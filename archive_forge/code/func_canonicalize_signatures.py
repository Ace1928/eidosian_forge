from absl import logging
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import attributes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.trackable import base
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def canonicalize_signatures(signatures):
    """Converts `signatures` into a dictionary of concrete functions."""
    if signatures is None:
        return ({}, {}, {})
    if not isinstance(signatures, collections_abc.Mapping):
        signatures = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signatures}
    num_normalized_signatures_counter = 0
    concrete_signatures = {}
    wrapped_functions = {}
    defaults = {}
    for signature_key, function in signatures.items():
        original_function = signature_function = _get_signature(function)
        if signature_function is None:
            raise ValueError(f'Expected a TensorFlow function for which to generate a signature, but got {function}. Only `tf.functions` with an input signature or concrete functions can be used as a signature.')
        wrapped_functions[original_function] = signature_function = wrapped_functions.get(original_function) or function_serialization.wrap_cached_variables(original_function)
        _validate_inputs(signature_function)
        if num_normalized_signatures_counter < _NUM_DISPLAY_NORMALIZED_SIGNATURES:
            signature_name_changes = _get_signature_name_changes(signature_function)
            if signature_name_changes:
                num_normalized_signatures_counter += 1
                logging.info('Function `%s` contains input name(s) %s with unsupported characters which will be renamed to %s in the SavedModel.', compat.as_str(signature_function.graph.name), ', '.join(signature_name_changes.keys()), ', '.join(signature_name_changes.values()))

        def signature_wrapper(**kwargs):
            structured_outputs = signature_function(**kwargs)
            return _normalize_outputs(structured_outputs, signature_function.name, signature_key)
        if hasattr(function, '__name__'):
            signature_wrapper.__name__ = 'signature_wrapper_' + function.__name__
        experimental_attributes = {}
        for attr in attributes.POLYMORPHIC_FUNCTION_ALLOWLIST:
            attr_value = signature_function.function_def.attr.get(attr, None)
            if attr != attributes.NO_INLINE and attr_value is not None:
                experimental_attributes[attr] = attr_value
        if not experimental_attributes:
            experimental_attributes = None
        wrapped_function = def_function.function(signature_wrapper, experimental_attributes=experimental_attributes)
        tensor_spec_signature = {}
        if signature_function.structured_input_signature is not None:
            inputs = filter(lambda x: isinstance(x, tensor.TensorSpec), nest.flatten(signature_function.structured_input_signature, expand_composites=True))
        else:
            inputs = signature_function.inputs
        for keyword, inp in zip(signature_function._arg_keywords, inputs):
            keyword = compat.as_str(keyword)
            if isinstance(inp, tensor.TensorSpec):
                spec = tensor.TensorSpec(inp.shape, inp.dtype, name=keyword)
            else:
                spec = tensor.TensorSpec.from_tensor(inp, name=keyword)
            tensor_spec_signature[keyword] = spec
        final_concrete = wrapped_function._get_concrete_function_garbage_collected(**tensor_spec_signature)
        if len(final_concrete._arg_keywords) == 1:
            final_concrete._num_positional_args = 1
        else:
            final_concrete._num_positional_args = 0
        concrete_signatures[signature_key] = final_concrete
        if isinstance(function, core.GenericFunction):
            flattened_defaults = nest.flatten(function.function_spec.fullargspec.defaults)
            len_default = len(flattened_defaults or [])
            arg_names = list(tensor_spec_signature.keys())
            if len_default > 0:
                for arg, default in zip(arg_names[-len_default:], flattened_defaults or []):
                    if not isinstance(default, tensor.Tensor):
                        continue
                    defaults.setdefault(signature_key, {})[arg] = default
    return (concrete_signatures, wrapped_functions, defaults)
import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def composite_tensor_variant_to_components(encoded: _atypes.TensorFuzzingAnnotation[_atypes.Variant], metadata: str, Tcomponents, name=None):
    """Decodes a `variant` scalar Tensor into an `ExtensionType` value.

  Returns the Tensor components encoded in a `CompositeTensorVariant`.

  Raises an error if `type_spec_proto` doesn't match the TypeSpec
  in `encoded`.

  Args:
    encoded: A `Tensor` of type `variant`.
      A scalar `variant` Tensor containing an encoded ExtensionType value.
    metadata: A `string`.
      String serialization for the TypeSpec.  Must be compatible with the
      `TypeSpec` contained in `encoded`.  (Note: the encoding for the TypeSpec
      may change in future versions of TensorFlow.)
    Tcomponents: A list of `tf.DTypes`. Expected dtypes for components.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tcomponents`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CompositeTensorVariantToComponents', name, encoded, 'metadata', metadata, 'Tcomponents', Tcomponents)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return composite_tensor_variant_to_components_eager_fallback(encoded, metadata=metadata, Tcomponents=Tcomponents, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    metadata = _execute.make_str(metadata, 'metadata')
    if not isinstance(Tcomponents, (list, tuple)):
        raise TypeError("Expected list for 'Tcomponents' argument to 'composite_tensor_variant_to_components' Op, not %r." % Tcomponents)
    Tcomponents = [_execute.make_type(_t, 'Tcomponents') for _t in Tcomponents]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CompositeTensorVariantToComponents', encoded=encoded, metadata=metadata, Tcomponents=Tcomponents, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('metadata', _op.get_attr('metadata'), 'Tcomponents', _op.get_attr('Tcomponents'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CompositeTensorVariantToComponents', _inputs_flat, _attrs, _result)
    return _result
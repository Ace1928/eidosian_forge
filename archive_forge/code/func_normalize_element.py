import collections
import functools
import itertools
import wrapt
from tensorflow.python.data.util import nest
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import internal
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.nest_util import CustomNestProtocol
from tensorflow.python.util.tf_export import tf_export
def normalize_element(element, element_signature=None):
    """Normalizes a nested structure of element components.

  * Components matching `SparseTensorSpec` are converted to `SparseTensor`.
  * Components matching `RaggedTensorSpec` are converted to `RaggedTensor`.
  * Components matching `VariableSpec` are converted to `Tensor`.
  * Components matching `DatasetSpec` or `TensorArraySpec` are passed through.
  * `CompositeTensor` components are passed through.
  * All other components are converted to `Tensor`.

  Args:
    element: A nested structure of individual components.
    element_signature: (Optional.) A nested structure of `tf.DType` objects
      corresponding to each component of `element`. If specified, it will be
      used to set the exact type of output tensor when converting input
      components which are not tensors themselves (e.g. numpy arrays, native
      python types, etc.)

  Returns:
    A nested structure of `Tensor`, `Variable`, `Dataset`, `SparseTensor`,
    `RaggedTensor`, or `TensorArray` objects.
  """
    normalized_components = []
    if element_signature is None:
        components = nest.flatten(element)
        flattened_signature = [None] * len(components)
        pack_as = element
    else:
        flattened_signature = nest.flatten(element_signature)
        components = nest.flatten_up_to(element_signature, element)
        pack_as = element_signature
    with ops.name_scope('normalize_element'):
        for i, (t, spec) in enumerate(zip(components, flattened_signature)):
            try:
                if spec is None:
                    spec = type_spec_from_value(t, use_fallback=False)
            except TypeError:
                normalized_components.append(ops.convert_to_tensor(t, name='component_%d' % i))
            else:
                if spec.__class__.__name__ == 'DatasetSpec':
                    normalized_components.append(t)
                elif isinstance(spec, sparse_tensor.SparseTensorSpec):
                    normalized_components.append(sparse_tensor.SparseTensor.from_value(t))
                elif isinstance(spec, ragged_tensor.RaggedTensorSpec):
                    normalized_components.append(ragged_tensor.convert_to_tensor_or_ragged_tensor(t, name='component_%d' % i))
                elif isinstance(spec, tensor_array_ops.TensorArraySpec):
                    normalized_components.append(t)
                elif isinstance(spec, NoneTensorSpec):
                    normalized_components.append(NoneTensor())
                elif isinstance(spec, resource_variable_ops.VariableSpec):
                    normalized_components.append(ops.convert_to_tensor(t, name=f'component_{i}', dtype=spec.dtype))
                elif isinstance(t, composite_tensor.CompositeTensor):
                    normalized_components.append(t)
                else:
                    dtype = getattr(spec, 'dtype', None)
                    normalized_components.append(ops.convert_to_tensor(t, name='component_%d' % i, dtype=dtype))
    return nest.pack_sequence_as(pack_as, normalized_components)
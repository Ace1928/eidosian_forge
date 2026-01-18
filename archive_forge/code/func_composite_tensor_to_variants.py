from tensorflow.core.protobuf import composite_tensor_variant_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_composite_tensor_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
def composite_tensor_to_variants(value, type_spec=None, name=None):
    """Encodes `value` as a scalar variant tensor.

  Args:
    value: The `ExtensionType` value to encode.
    type_spec: Information about the value's type that should be included in the
      encoding.
    name: Optional name for the operation.

  Returns:
    A Tensor with shape=`()` and dtype=`tf.variant`.

  Raises:
    ValueError: If `type_spec` is not compatible with `value`.
  """
    if not isinstance(value, composite_tensor.CompositeTensor):
        raise TypeError(f'Expected `value` to be a CompositeTensor. Received {type(value)}.')
    if type_spec is None:
        type_spec = value._type_spec
    if not type_spec.is_compatible_with(value):
        raise ValueError(f'`type_spec` {type_spec} is not compatible with `value` {value!r}.')
    metadata = composite_tensor_variant_pb2.CompositeTensorVariantMetadata()
    metadata.type_spec_proto.CopyFrom(nested_structure_coder.encode_structure(type_spec).type_spec_value)
    return gen_composite_tensor_ops.CompositeTensorVariantFromComponents(components=nest.flatten(value, expand_composites=True), metadata=metadata.SerializeToString(), name=name)
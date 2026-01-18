from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import sparse_ops
def serialize_many_sparse_tensors(tensors):
    """Serializes many sparse tensors into a batch.

  Args:
    tensors: a tensor structure to serialize.

  Returns:
    `tensors` with any sparse tensors replaced by the serialized batch.
  """
    ret = nest.pack_sequence_as(tensors, [sparse_ops.serialize_many_sparse(tensor, out_type=dtypes.variant) if sparse_tensor.is_sparse(tensor) else tensor for tensor in nest.flatten(tensors)])
    return ret
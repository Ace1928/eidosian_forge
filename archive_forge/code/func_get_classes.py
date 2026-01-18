from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import sparse_ops
def get_classes(tensors):
    """Gets classes for a structure of tensors.

  Args:
    tensors: the tensor structure to get classes for.

  Returns:
    a structure matching the nested structure of `tensors`, containing
    `tf.sparse.SparseTensor` at positions where `tensors` contains a sparse
    tensor and `tf.Tensor` otherwise.
  """
    return nest.pack_sequence_as(tensors, [sparse_tensor.SparseTensor if isinstance(tensor, sparse_tensor.SparseTensor) else tensor_lib.Tensor for tensor in nest.flatten(tensors)])
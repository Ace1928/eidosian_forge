from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import sparse_ops
def any_sparse(classes):
    """Checks for sparse tensor.

  Args:
    classes: a structure of objects that identify the dataset item classes

  Returns:
    `True` if `classes` contains a sparse tensor type and `False` otherwise.
  """
    return any((c is sparse_tensor.SparseTensor for c in nest.flatten(classes)))
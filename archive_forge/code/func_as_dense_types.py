from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import sparse_ops
def as_dense_types(types, classes):
    """Converts sparse tensor types to `dtypes.variant`.

  Args:
    types: a structure of types to convert.
    classes: a structure of objects that identify the dataset item classes

  Returns:
    a structure matching the nested structure of `types`, containing
    `dtypes.variant` at positions where `classes` contains
    `tf.sparse.SparseTensor` and matching contents of `types` otherwise
  """
    ret = nest.pack_sequence_as(types, [dtypes.variant if c is sparse_tensor.SparseTensor else ty for ty, c in zip(nest.flatten(types), nest.flatten(classes))])
    return ret
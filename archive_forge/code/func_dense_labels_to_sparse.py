import uuid
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def dense_labels_to_sparse(dense, length):
    """Convert dense labels with sequence lengths to sparse tensor.

  Args:
    dense: tensor of shape [batch, max_length]
    length: int tensor of shape [batch] The length of each sequence in dense.

  Returns:
    tf.sparse.SparseTensor with values only for the valid elements of sequences.
  """
    flat_values = array_ops.reshape(dense, [-1])
    flat_indices = math_ops.range(array_ops.shape(flat_values, out_type=dtypes.int64)[0])
    mask = array_ops.sequence_mask(length, maxlen=array_ops.shape(dense)[1])
    flat_mask = array_ops.reshape(mask, [-1])
    indices = array_ops.expand_dims(array_ops.boolean_mask(flat_indices, flat_mask), 1)
    values = array_ops.boolean_mask(flat_values, flat_mask)
    sparse = sparse_tensor.SparseTensor(indices=indices, values=math_ops.cast(values, dtypes.int32), dense_shape=array_ops.shape(flat_values, out_type=dtypes.int64))
    reshaped = sparse_ops.sparse_reshape(sparse, array_ops.shape(dense))
    max_length = math_ops.reduce_max(length)
    return sparse_tensor.SparseTensor(indices=reshaped.indices, values=reshaped.values, dense_shape=[math_ops.cast(reshaped.dense_shape[0], dtypes.int64), math_ops.cast(max_length, dtypes.int64)])
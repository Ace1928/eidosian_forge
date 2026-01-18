from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import tensor_array_ops
def is_range_tensor(t):
    """Returns True if a tensor is the result of a tf.range op. Best effort."""
    return tensor_util.is_tf_type(t) and hasattr(t, 'op') and (t.op.type == 'Range')
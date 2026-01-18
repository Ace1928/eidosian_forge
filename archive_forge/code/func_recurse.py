from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def recurse(v):
    return _replace_ragged_with_flat_values(v, partition_lists, flat_values_nrows)
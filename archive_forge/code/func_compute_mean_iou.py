from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def compute_mean_iou(_, total_cm):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = math_ops.cast(math_ops.reduce_sum(total_cm, 0), dtypes.float32)
    sum_over_col = math_ops.cast(math_ops.reduce_sum(total_cm, 1), dtypes.float32)
    cm_diag = math_ops.cast(array_ops.diag_part(total_cm), dtypes.float32)
    denominator = sum_over_row + sum_over_col - cm_diag
    num_valid_entries = math_ops.reduce_sum(math_ops.cast(math_ops.not_equal(denominator, 0), dtype=dtypes.float32))
    denominator = array_ops.where(math_ops.greater(denominator, 0), denominator, array_ops.ones_like(denominator))
    iou = math_ops.divide(cm_diag, denominator)
    result = array_ops.where(math_ops.greater(num_valid_entries, 0), math_ops.reduce_sum(iou, name='mean_iou') / num_valid_entries, 0)
    return result
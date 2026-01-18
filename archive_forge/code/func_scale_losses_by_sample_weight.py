from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
def scale_losses_by_sample_weight(losses, sample_weight):
    """Scales loss values by the given sample weights.

  `sample_weight` dimensions are updated to match with the dimension of `losses`
  if possible by using squeeze/expand/broadcast.

  Args:
    losses: Loss tensor.
    sample_weight: Sample weights tensor.

  Returns:
    `losses` scaled by `sample_weight` with dtype float32.
  """
    losses = math_ops.cast(losses, dtypes.float32)
    sample_weight = math_ops.cast(sample_weight, dtypes.float32)
    losses, _, sample_weight = squeeze_or_expand_dimensions(losses, None, sample_weight)
    return math_ops.multiply(losses, sample_weight)
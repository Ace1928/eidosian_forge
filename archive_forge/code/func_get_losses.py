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
@tf_export(v1=['losses.get_losses'])
def get_losses(scope=None, loss_collection=ops.GraphKeys.LOSSES):
    """Gets the list of losses from the loss_collection.

  Args:
    scope: An optional scope name for filtering the losses to return.
    loss_collection: Optional losses collection.

  Returns:
    a list of loss tensors.
  """
    return ops.get_collection(loss_collection, scope)
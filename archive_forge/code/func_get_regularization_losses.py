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
@tf_export(v1=['losses.get_regularization_losses'])
def get_regularization_losses(scope=None):
    """Gets the list of regularization losses.

  Args:
    scope: An optional scope name for filtering the losses to return.

  Returns:
    A list of regularization losses as Tensors.
  """
    return ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES, scope)
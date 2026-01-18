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
@tf_export(v1=['losses.add_loss'])
def add_loss(loss, loss_collection=ops.GraphKeys.LOSSES):
    """Adds a externally defined loss to the collection of losses.

  Args:
    loss: A loss `Tensor`.
    loss_collection: Optional collection to add the loss to.
  """
    if loss_collection and (not context.executing_eagerly()):
        ops.add_to_collection(loss_collection, loss)
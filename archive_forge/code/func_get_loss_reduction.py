from collections import abc
import contextlib
import threading
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_values as tpu_values_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['distribute.get_loss_reduction'])
def get_loss_reduction():
    """`tf.distribute.ReduceOp` corresponding to the last loss reduction.

  This is used to decide whether loss should be scaled in optimizer (used only
  for estimator + v1 optimizer use case).

  Returns:
    `tf.distribute.ReduceOp` corresponding to the last loss reduction for
    estimator and v1 optimizer use case. `tf.distribute.ReduceOp.SUM` otherwise.
  """
    if not distribute_lib.get_strategy()._scale_loss_for_estimator:
        return ReduceOp.SUM
    last_reduction = ops.get_default_graph()._last_loss_reduction
    if last_reduction == losses_impl.Reduction.SUM or last_reduction == 'sum':
        return ReduceOp.SUM
    return ReduceOp.MEAN
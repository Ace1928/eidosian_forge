from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import base_delegate
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.util import nest
def get_scaled_loss(self, loss):
    """Scales the loss by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to scale the loss before
    passing the loss to `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_unscaled_gradients` should also be called.
    See the `tf.keras.mixed_precision.LossScaleOptimizer` doc for
    an example.

    Args:
      loss: The loss, which will be multiplied by the loss scale. Can either be
        a tensor or a callable returning a tensor.

    Returns:
      `loss` multiplied by `LossScaleOptimizer.loss_scale`.
    """
    if callable(loss):

        def new_loss():
            loss_val = loss()
            return loss_val * math_ops.cast(self.loss_scale, loss_val.dtype)
        return new_loss
    else:
        return loss * math_ops.cast(self.loss_scale, loss.dtype)
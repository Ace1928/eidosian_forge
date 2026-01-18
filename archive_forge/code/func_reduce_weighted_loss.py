from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
def reduce_weighted_loss(weighted_losses, reduction=ReductionV2.SUM_OVER_BATCH_SIZE):
    """Reduces the individual weighted loss measurements."""
    if reduction == ReductionV2.NONE:
        loss = weighted_losses
    else:
        loss = math_ops.reduce_sum(weighted_losses)
        if reduction == ReductionV2.SUM_OVER_BATCH_SIZE:
            loss = _safe_mean(loss, _num_elements(weighted_losses))
    return loss
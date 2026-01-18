import abc
import functools
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.tools.docs import doc_controls
@dispatch.add_dispatch_support
def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0, axis=-1):
    """Computes the categorical crossentropy loss.

  Standalone usage:

  >>> y_true = [[0, 1, 0], [0, 0, 1]]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> loss.numpy()
  array([0.0513, 2.303], dtype=float32)

  Args:
    y_true: Tensor of one-hot true targets.
    y_pred: Tensor of predicted targets.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
      example, if `0.1`, use `0.1 / num_classes` for non-target labels
      and `0.9 + 0.1 / num_classes` for target labels.
    axis: Defaults to -1. The dimension along which the entropy is
      computed.

  Returns:
    Categorical crossentropy loss value.
  """
    y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    label_smoothing = tensor_conversion.convert_to_tensor_v2_with_dispatch(label_smoothing, dtype=backend.floatx())

    def _smooth_labels():
        num_classes = math_ops.cast(array_ops.shape(y_true)[-1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
    y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y_true)
    return backend.categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis)
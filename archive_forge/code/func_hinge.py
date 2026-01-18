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
def hinge(y_true, y_pred):
    """Computes the hinge loss between `y_true` and `y_pred`.

  `loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.choice([-1, 1], size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.hinge(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(),
  ...     np.mean(np.maximum(1. - y_true * y_pred, 0.), axis=-1))

  Args:
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
      If binary (0 or 1) labels are provided they will be converted to -1 or 1.
      shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
  """
    y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = _maybe_convert_labels(y_true)
    return backend.mean(math_ops.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)
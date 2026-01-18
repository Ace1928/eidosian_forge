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
def mean_absolute_percentage_error(y_true, y_pred):
    """Computes the mean absolute percentage error between `y_true` and `y_pred`.

  `loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.random(size=(2, 3))
  >>> y_true = np.maximum(y_true, 1e-7)  # Prevent division by zero
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(),
  ...     100. * np.mean(np.abs((y_true - y_pred) / y_true), axis=-1))

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`.
  """
    y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.abs((y_true - y_pred) / backend.maximum(math_ops.abs(y_true), backend.epsilon()))
    return 100.0 * backend.mean(diff, axis=-1)
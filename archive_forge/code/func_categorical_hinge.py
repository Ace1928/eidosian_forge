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
def categorical_hinge(y_true, y_pred):
    """Computes the categorical hinge loss between `y_true` and `y_pred`.

  `loss = maximum(neg - pos + 1, 0)`
  where `neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)`

  Standalone usage:

  >>> y_true = np.random.randint(0, 3, size=(2,))
  >>> y_true = tf.keras.utils.to_categorical(y_true, num_classes=3)
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.categorical_hinge(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> pos = np.sum(y_true * y_pred, axis=-1)
  >>> neg = np.amax((1. - y_true) * y_pred, axis=-1)
  >>> assert np.array_equal(loss.numpy(), np.maximum(0., neg - pos + 1.))

  Args:
    y_true: The ground truth values. `y_true` values are expected to be
    either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor).
    y_pred: The predicted values.

  Returns:
    Categorical hinge loss values.
  """
    y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    pos = math_ops.reduce_sum(y_true * y_pred, axis=-1)
    neg = math_ops.reduce_max((1.0 - y_true) * y_pred, axis=-1)
    zero = math_ops.cast(0.0, y_pred.dtype)
    return math_ops.maximum(neg - pos + 1.0, zero)
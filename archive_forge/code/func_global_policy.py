import contextlib
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.mixed_precision import device_compatibility_check
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.experimental import mixed_precision_global_state
def global_policy():
    """Returns the global dtype policy.

  The global policy is the default `tf.keras.mixed_precision.Policy` used for
  layers, if no policy is passed to the layer constructor. If no policy has been
  set with `keras.mixed_precision.set_global_policy`, this will return a policy
  constructed from `tf.keras.backend.floatx()` (floatx defaults to float32).

  >>> tf.keras.mixed_precision.global_policy()
  <Policy "float32">
  >>> tf.keras.layers.Dense(10).dtype_policy  # Defaults to the global policy
  <Policy "float32">

  If TensorFlow 2 behavior has been disabled with
  `tf.compat.v1.disable_v2_behavior()`, this will instead return a special
  "_infer" policy which infers the dtype from the dtype of the first input the
  first time the layer is called. This behavior matches the behavior that
  existed in TensorFlow 1.

  See `tf.keras.mixed_precision.Policy` for more information on policies.

  Returns:
    The global Policy.
  """
    if _global_policy is None:
        if base_layer_utils.v2_dtype_behavior_enabled():
            return Policy(backend.floatx())
        else:
            return Policy('_infer')
    return _global_policy
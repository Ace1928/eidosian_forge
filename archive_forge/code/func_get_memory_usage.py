from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, "Use tf.config.experimental.get_memory_info(device)['current'] instead.")
@tf_export('config.experimental.get_memory_usage')
def get_memory_usage(device):
    """Get the current memory usage, in bytes, for the chosen device.

  This function is deprecated in favor of
  `tf.config.experimental.get_memory_info`. Calling this function is equivalent
  to calling `tf.config.experimental.get_memory_info()['current']`.

  See https://www.tensorflow.org/api_docs/python/tf/device for specifying device
  strings.

  For example:

  >>> gpu_devices = tf.config.list_physical_devices('GPU')
  >>> if gpu_devices:
  ...   tf.config.experimental.get_memory_usage('GPU:0')

  Does not work for CPU.

  For GPUs, TensorFlow will allocate all the memory by default, unless changed
  with `tf.config.experimental.set_memory_growth`. This function only returns
  the memory that TensorFlow is actually using, not the memory that TensorFlow
  has allocated on the GPU.

  Args:
    device: Device string to get the bytes in use for, e.g. `"GPU:0"`

  Returns:
    Total memory usage in bytes.

  Raises:
    ValueError: Non-existent or CPU device specified.
  """
    return get_memory_info(device)['current']
import contextlib
import threading
from typing import Any, Callable, Optional, Sequence
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.device_name', v1=[])
def device_name() -> str:
    """Returns the singleton DTensor device's name.

  This function can be used in the following way:

  ```python
  import tensorflow as tf

  with tf.device(dtensor.device_name()):
    # ...
  ```
  """
    return _dtensor_device().name
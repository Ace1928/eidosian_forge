import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.preferred_device_type', v1=[])
def preferred_device_type() -> str:
    """Returns the preferred device type for the accelerators.

  The returned device type is determined by checking the first present device
  type from all supported device types in the order of 'TPU', 'GPU', 'CPU'.
  """
    if is_tpu_present():
        return 'TPU'
    elif is_gpu_present():
        return 'GPU'
    return 'CPU'
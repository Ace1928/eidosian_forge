import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
def gpu_use_nccl_communication() -> bool:
    """Return True if environment indicates NCCL shall be used for GPU."""
    return os.environ.get('DTENSOR_GPU_USE_NCCL_COMMUNICATION', '0') != '0'
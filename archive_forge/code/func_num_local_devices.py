import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.num_local_devices', v1=[])
def num_local_devices(device_type: str) -> int:
    """Returns the number of devices of device_type configured on this client."""
    if device_type.upper() in ['CPU', 'GPU']:
        context_config = context.get_config()
        return context_config.device_count[device_type.upper()]
    return len(tf_config.list_physical_devices(device_type))
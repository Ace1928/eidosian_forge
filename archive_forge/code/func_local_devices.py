import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.local_devices', v1=[])
def local_devices(device_type: str, for_client_id: Optional[int]=None) -> List[tf_device.DeviceSpec]:
    """Returns a list of device specs configured on this client."""
    if device_type.upper() not in ['CPU', 'GPU', 'TPU']:
        raise ValueError(f'Device type {device_type} is not CPU, GPU, or TPU.')
    if for_client_id is None:
        for_client_id = client_id()
    return [tf_device.DeviceSpec(job=job_name(), replica=0, task=for_client_id, device_type=device_type, device_index=i) for i in range(num_local_devices(device_type))]
import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.client_id', v1=[])
def client_id() -> int:
    """Returns this client's ID."""
    client_id_value = int(os.environ.get(_DT_CLIENT_ID, '0'))
    if client_id_value < 0:
        raise ValueError(f'Environment variable {_DT_CLIENT_ID} must be >= 0, got {client_id_value}. ')
    if client_id_value >= num_clients():
        raise ValueError(f'Environment variable {_DT_CLIENT_ID} must be < {num_clients()}, got {client_id_value}')
    return client_id_value
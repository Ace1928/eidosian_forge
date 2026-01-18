import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.full_job_name', v1=[])
def full_job_name(task_id: Optional[int]=None) -> str:
    """Returns the fully qualified TF job name for this or another task."""
    if task_id is None:
        task_id = client_id()
    if num_clients() == 1 and task_id != 0:
        raise ValueError(f'Unexpected task ID {task_id} in local runs')
    return f'{job_name()}/replica:0/task:{task_id}'
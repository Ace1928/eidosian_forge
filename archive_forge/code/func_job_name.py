import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.job_name', v1=[])
def job_name() -> str:
    """Returns the job name used by all clients in this DTensor cluster."""
    return os.environ.get(_DT_JOB_NAME, 'localhost' if num_clients() == 1 else 'worker')
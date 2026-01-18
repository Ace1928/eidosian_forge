import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
def backend_is_pw() -> bool:
    """Return True if environment indicates the backend is Pathways."""
    return os.environ.get('DTENSOR_USE_PARALLEL_EXECUTOR') == 'pw'
from typing import List, Optional
from absl import logging
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.platform import remote_utils
from tensorflow.python.util.tf_export import tf_export
def set_initialized(value):
    """Sets if accelerator system has been initialized."""
    global _INITIALIZED_ACCELERATOR_SYSTEM_TYPE
    _INITIALIZED_ACCELERATOR_SYSTEM_TYPE = value
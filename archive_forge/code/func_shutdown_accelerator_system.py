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
@tf_export('experimental.dtensor.shutdown_accelerator_system', 'experimental.dtensor.shutdown_tpu_system', v1=[])
def shutdown_accelerator_system() -> None:
    """Shuts down the accelerator system."""
    global _INITIALIZED_ACCELERATOR_SYSTEM_TYPE
    try:
        context.async_wait()
    finally:
        if not is_initialized():
            raise ValueError('Accelerator system is not initialized. Call tf.experimental.dtensor.initialize_accelerator_system first.')
        device_type = _INITIALIZED_ACCELERATOR_SYSTEM_TYPE
        if not config.is_local_mode():
            raise ValueError('Shutting down accelerator system under multi-client mode is not supported.')
        if device_type == 'TPU' and (not config.backend_is_pw()):
            tpu_util.shutdown_tpu_system()
        context._reset_context()
        context.context()._clear_caches()
        _INITIALIZED_ACCELERATOR_SYSTEM_TYPE = None
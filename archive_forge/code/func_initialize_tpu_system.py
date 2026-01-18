import functools
import time
from typing import List, Optional, Dict
import numpy as np
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.util.tf_export import tf_export
def initialize_tpu_system():
    """Initializes the TPU system."""
    context.ensure_initialized()
    context.async_wait()
    context.context()._clear_caches()
    use_tfrt_host_runtime = context.context().use_tfrt
    logging.info('Using TFRT host runtime is set to %s', use_tfrt_host_runtime)
    try:
        task_id = config.client_id()
        num_tasks = config.num_clients()
        num_devices = config.num_global_devices(_TPU_DEVICE_TYPE)
        tpu_topology, device = tpu_system_init_helper(task_id, num_tasks, num_devices, use_tfrt_host_runtime=use_tfrt_host_runtime)
        global _tpu_topology
        _tpu_topology = tpu_topology
        logging.vlog(1, 'TPU Topology: %s, %s', tpu_topology.mesh_shape, tpu_topology.device_coordinates)
        global _dtensor_device
        _dtensor_device = device
        context.async_wait()
    except errors.InvalidArgumentError as e:
        raise errors.NotFoundError(None, None, 'Initialization failed, no valid TPUs found. ' + str(e)) from e
    except errors.InternalError as e:
        logging.error('Hit internal error during TPU system initialization. ' + 'It is likely hareware failure. \nPlease check the error ' + "messages above to see whether that's the case. \nIf so, " + 'consider to restart the job or try another machine.')
        raise e
    logging.info('Clearing out eager caches')
    context.context()._clear_caches()
import collections
import contextlib
import copy
import gc
import itertools
import os
import random
import threading
from absl import logging
import numpy as np
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import execute
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def set_logical_cpu_devices(self, num_cpus, prefix=''):
    """Set virtual CPU devices in context.

    If virtual CPU devices are already configured at context initialization
    by tf.config.set_logical_device_configuration(), this method should not be
    called.

    Args:
      num_cpus: Number of virtual CPUs.
      prefix: Device name prefix.

    Raises:
     RuntimeError: If virtual CPUs are already configured at context
     initialization.
    """
    server_def = self._server_def or self._collective_ops_server_def
    local_prefix = ['/device']
    if server_def is not None:
        local_prefix.append('/job:%s/replica:0/task:%d' % (server_def.job_name, server_def.task_index))
    logical_local_devices = [d for d in self.list_logical_devices('CPU') if d.name.startswith(tuple(local_prefix))]
    self.ensure_initialized()
    if len(logical_local_devices) > 1:
        raise RuntimeError('Virtual CPUs already set, cannot modify again.')
    pywrap_tfe.TFE_SetLogicalCpuDevices(self._context_handle, num_cpus, prefix)
    self._initialize_logical_devices()
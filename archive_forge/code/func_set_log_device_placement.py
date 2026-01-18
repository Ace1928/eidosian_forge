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
@tf_export('debugging.set_log_device_placement')
def set_log_device_placement(enabled):
    """Turns logging for device placement decisions on or off.

  Operations execute on a particular device, producing and consuming tensors on
  that device. This may change the performance of the operation or require
  TensorFlow to copy data to or from an accelerator, so knowing where operations
  execute is useful for debugging performance issues.

  For more advanced profiling, use the [TensorFlow
  profiler](https://www.tensorflow.org/guide/profiler).

  Device placement for operations is typically controlled by a `tf.device`
  scope, but there are exceptions, for example operations on a `tf.Variable`
  which follow the initial placement of the variable. Turning off soft device
  placement (with `tf.config.set_soft_device_placement`) provides more explicit
  control.

  >>> tf.debugging.set_log_device_placement(True)
  >>> tf.ones([])
  >>> # [...] op Fill in device /job:localhost/replica:0/task:0/device:GPU:0
  >>> with tf.device("CPU"):
  ...  tf.ones([])
  >>> # [...] op Fill in device /job:localhost/replica:0/task:0/device:CPU:0
  >>> tf.debugging.set_log_device_placement(False)

  Turning on `tf.debugging.set_log_device_placement` also logs the placement of
  ops inside `tf.function` when the function is called.

  Args:
    enabled: Whether to enabled device placement logging.
  """
    context().log_device_placement = enabled
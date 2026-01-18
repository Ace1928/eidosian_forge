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
def set_memory_growth(self, dev, enable):
    """Set if memory growth should be enabled for a PhysicalDevice."""
    self._initialize_physical_devices()
    if dev not in self._physical_devices:
        raise ValueError('Unrecognized device: %s' % repr(dev))
    if dev in self._virtual_device_map:
        raise ValueError('Cannot set memory growth on device when virtual devices configured')
    if dev.device_type != 'GPU' and dev not in self._pluggable_devices:
        raise ValueError('Cannot set memory growth on non-GPU and non-Pluggable devices')
    if self._memory_growth_map.get(dev) == enable:
        return
    if self._context_handle is not None:
        raise RuntimeError('Physical devices cannot be modified after being initialized')
    self._memory_growth_map[dev] = enable
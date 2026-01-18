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
def get_device_details(self, device):
    """Returns details about a physical devices.

    Args:
      device: A `tf.config.PhysicalDevice` returned by
        `tf.config.list_physical_devices` or `tf.config.get_visible_devices`.

    Returns:
      A dict with string keys.
    """
    if not isinstance(device, PhysicalDevice):
        raise ValueError('device must be a tf.config.PhysicalDevice, but got: %s' % (device,))
    if self._physical_device_to_index is None or device not in self._physical_device_to_index:
        raise ValueError('The PhysicalDevice must be one obtained from calling `tf.config.list_physical_devices`, but got: %s' % (device,))
    index = self._physical_device_to_index[device]
    details = pywrap_tfe.TF_GetDeviceDetails(index)
    if 'compute_capability' in details:
        try:
            major, minor = details['compute_capability'].split('.')
            details['compute_capability'] = (int(major), int(minor))
        except ValueError:
            raise RuntimeError('Device returned compute capability an in invalid format: %s' % details['compute_capability'])
    return details
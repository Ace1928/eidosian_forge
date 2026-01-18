import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def device_path_to_device_name(device_dir):
    """Parse device name from device path.

  Args:
    device_dir: (str) a directory name for the device.

  Returns:
    (str) parsed device name.
  """
    path_items = os.path.basename(device_dir)[len(METADATA_FILE_PREFIX) + len(DEVICE_TAG):].split(',')
    return '/'.join([path_item.replace('device_', 'device:').replace('_', ':', 1) for path_item in path_items])
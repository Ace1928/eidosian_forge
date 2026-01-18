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
def get_rel_timestamps(self, node_name, output_slot, debug_op, device_name=None):
    """Get the relative timestamp from for a debug-dumped tensor.

    Relative timestamp means (absolute timestamp - `t0`), where `t0` is the
    absolute timestamp of the first dumped tensor in the dump root. The tensor
    may be dumped multiple times in the dump root directory, so a list of
    relative timestamps (`numpy.ndarray`) is returned.

    Args:
      node_name: (`str`) name of the node that the tensor is produced by.
      output_slot: (`int`) output slot index of tensor.
      debug_op: (`str`) name of the debug op.
      device_name: (`str`) name of the device. If there is only one device or if
        the specified debug_watch_key exists on only one device, this argument
        is optional.

    Returns:
      (`list` of `int`) list of relative timestamps.

    Raises:
      WatchKeyDoesNotExistInDebugDumpDirError: If the tensor watch key does not
        exist in the debug dump data.
    """
    device_name = self._infer_device_name(device_name, node_name)
    watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
    if watch_key not in self._watch_key_to_datum[device_name]:
        raise WatchKeyDoesNotExistInDebugDumpDirError('Watch key "%s" does not exist in the debug dump' % watch_key)
    return self._watch_key_to_rel_time[device_name][watch_key]
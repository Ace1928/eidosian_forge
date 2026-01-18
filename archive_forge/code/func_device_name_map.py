import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def device_name_map(self):
    """Get a map mapping device IDs to device names."""
    return {device_id: self._device_by_id[device_id].device_name for device_id in self._device_by_id}
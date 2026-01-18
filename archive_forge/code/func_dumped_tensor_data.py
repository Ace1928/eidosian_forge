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
@property
def dumped_tensor_data(self):
    """Retrieve dumped tensor data."""
    if len(self.devices()) == 1:
        return self._dump_tensor_data[self.devices()[0]]
    else:
        all_devices_data = self._dump_tensor_data.values()
        data = []
        for device_data in all_devices_data:
            data.extend(device_data)
        return sorted(data, key=lambda x: x.extended_timestamp)
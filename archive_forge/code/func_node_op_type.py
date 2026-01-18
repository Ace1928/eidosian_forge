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
def node_op_type(self, node_name, device_name=None):
    """Get the op type of given node.

    Args:
      node_name: (`str`) name of the node.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      (`str`) op type of the node.

    Raises:
      LookupError: If node op types have not been loaded
         from partition graphs yet.
    """
    if not self._debug_graphs:
        raise LookupError('Node op types are not loaded from partition graphs yet.')
    device_name = self._infer_device_name(device_name, node_name)
    return self._debug_graphs[device_name].node_op_types[node_name]
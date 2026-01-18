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
def node_attributes(self, node_name, device_name=None):
    """Get the attributes of a node.

    Args:
      node_name: Name of the node in question.
      device_name: (`str`) name of the device. If there is only one device or if
        node_name exists on only one device, this argument is optional.

    Returns:
      Attributes of the node.

    Raises:
      LookupError: If no partition graphs have been loaded.
    """
    if not self._debug_graphs:
        raise LookupError('No partition graphs have been loaded.')
    device_name = self._infer_device_name(device_name, node_name)
    return self._debug_graphs[device_name].node_attributes[node_name]
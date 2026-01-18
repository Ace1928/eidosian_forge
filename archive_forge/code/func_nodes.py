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
def nodes(self, device_name=None):
    """Get a list of all nodes from the partition graphs.

    Args:
      device_name: (`str`) name of device. If None, all nodes from all available
        devices will be included.

    Returns:
      All nodes' names, as a list of str.

    Raises:
      LookupError: If no partition graphs have been loaded.
      ValueError: If specified node name does not exist.
    """
    if not self._debug_graphs:
        raise LookupError('No partition graphs have been loaded.')
    if device_name is None:
        nodes = []
        for device_name in self._debug_graphs:
            nodes.extend(self._debug_graphs[device_name].node_inputs.keys())
        return nodes
    else:
        if device_name not in self._debug_graphs:
            raise ValueError('Invalid device name: %s' % device_name)
        return self._debug_graphs[device_name].node_inputs.keys()
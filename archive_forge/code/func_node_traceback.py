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
def node_traceback(self, element_name):
    """Try to retrieve the Python traceback of node's construction.

    Args:
      element_name: (`str`) Name of a graph element (node or tensor).

    Returns:
      (list) The traceback list object as returned by the `extract_trace`
        method of Python's traceback module.

    Raises:
      LookupError: If Python graph is not available for traceback lookup.
      KeyError: If the node cannot be found in the Python graph loaded.
    """
    if self._python_graph is None:
        raise LookupError('Python graph is not available for traceback lookup')
    node_name = debug_graphs.get_node_name(element_name)
    if node_name not in self._node_traceback:
        raise KeyError('Cannot find node "%s" in Python graph' % node_name)
    return self._node_traceback[node_name]
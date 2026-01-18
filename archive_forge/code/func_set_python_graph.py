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
def set_python_graph(self, python_graph):
    """Provide Python `Graph` object to the wrapper.

    Unlike the partition graphs, which are protobuf `GraphDef` objects, `Graph`
    is a Python object and carries additional information such as the traceback
    of the construction of the nodes in the graph.

    Args:
      python_graph: (ops.Graph) The Python Graph object.
    """
    self._python_graph = python_graph
    self._node_traceback = {}
    if self._python_graph:
        for op in self._python_graph.get_operations():
            self._node_traceback[op.name] = tuple(map(tuple, op.traceback))
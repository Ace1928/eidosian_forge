import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def add_inner_graph_id(self, inner_graph_id):
    """Add the debugger-generated ID of a graph nested within this graph.

    Args:
      inner_graph_id: The debugger-generated ID of the nested inner graph.
    """
    assert isinstance(inner_graph_id, str)
    self._inner_graph_ids.append(inner_graph_id)
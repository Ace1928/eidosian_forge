import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def graph_by_id(self, graph_id):
    """Get a DebuggedGraph object by its ID."""
    return self._graph_by_id[graph_id]
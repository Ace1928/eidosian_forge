import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def read_graph_execution_trace(self, graph_execution_trace_digest):
    """Read the detailed graph execution trace.

    Args:
      graph_execution_trace_digest: A `GraphExecutionTraceDigest` object.

    Returns:
      The corresponding `GraphExecutionTrace` object.
    """
    debug_event = self._reader.read_graph_execution_traces_event(graph_execution_trace_digest.locator)
    return self._graph_execution_trace_from_debug_event_proto(debug_event, graph_execution_trace_digest.locator)
import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def add_op(self, graph_op_creation_digest):
    """Add an op creation data object.

    Args:
      graph_op_creation_digest: A GraphOpCreationDigest data object describing
        the creation of an op inside this graph.
    """
    if graph_op_creation_digest.op_name in self._op_by_name:
        raise ValueError('Duplicate op name: %s (op type: %s)' % (graph_op_creation_digest.op_name, graph_op_creation_digest.op_type))
    self._op_by_name[graph_op_creation_digest.op_name] = graph_op_creation_digest
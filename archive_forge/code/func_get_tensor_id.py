import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def get_tensor_id(self, op_name, output_slot):
    """Get the ID of a symbolic tensor in this graph."""
    return self._op_by_name[op_name].output_tensor_ids[output_slot]
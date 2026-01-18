import collections
import errno
import functools
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import portpicker
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_server
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.util import compat
def on_value_event(self, event):
    """Implementation of the tensor value-carrying Event proto callback.

    Writes the Event proto to the file system for testing. The path written to
    follows the same pattern as the file:// debug URLs of tfdbg, i.e., the
    name scope of the op becomes the directory structure under the dump root
    directory.

    Args:
      event: The Event proto carrying a tensor value.

    Returns:
      If the debug node belongs to the set of currently activated breakpoints,
      a `EventReply` proto will be returned.
    """
    if self._dump_dir:
        self._write_value_event(event)
    else:
        value = event.summary.value[0]
        tensor_value = debug_data.load_tensor_from_event(event)
        self._event_listener_servicer.debug_tensor_values[value.node_name].append(tensor_value)
        items = event.summary.value[0].node_name.split(':')
        node_name = items[0]
        output_slot = int(items[1])
        debug_op = items[2]
        if (node_name, output_slot, debug_op) in self._event_listener_servicer.breakpoints:
            return debug_service_pb2.EventReply()
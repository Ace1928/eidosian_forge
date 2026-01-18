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
def on_graph_def(self, graph_def, device_name, wall_time):
    """Implementation of the tensor value-carrying Event proto callback.

    Args:
      graph_def: A GraphDef object.
      device_name: Name of the device on which the graph was created.
      wall_time: An epoch timestamp (in microseconds) for the graph.
    """
    if self._dump_dir:
        if self._grpc_path:
            self._write_graph_def(graph_def, device_name, wall_time)
        else:
            self._cached_graph_defs.append(graph_def)
            self._cached_graph_def_device_names.append(device_name)
            self._cached_graph_def_wall_times.append(wall_time)
    else:
        self._event_listener_servicer.partition_graph_defs.append(graph_def)
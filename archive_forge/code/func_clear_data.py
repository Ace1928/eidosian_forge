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
def clear_data(self):
    self.core_metadata_json_strings = []
    self.partition_graph_defs = []
    self.debug_tensor_values = collections.defaultdict(list)
    self._call_types = []
    self._call_keys = []
    self._origin_stacks = []
    self._origin_id_to_strings = []
    self._graph_tracebacks = []
    self._graph_versions = []
    self._source_files = []
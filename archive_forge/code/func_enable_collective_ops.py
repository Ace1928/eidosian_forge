import collections
import contextlib
import copy
import gc
import itertools
import os
import random
import threading
from absl import logging
import numpy as np
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import execute
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def enable_collective_ops(self, server_def):
    """Enable distributed collective ops with an appropriate server_def.

    Args:
      server_def: A tensorflow::ServerDef proto. Enables execution on remote
        devices.

    Raises:
      ValueError: if server_def is None.
      RuntimeError: if this method is not called at program startup.
    """
    if not server_def:
        raise ValueError('server_def is None.')
    self._collective_ops_server_def = server_def
    if self._context_handle is not None:
        logging.warning('Enabling collective ops after program startup may cause error when accessing previously created tensors.')
        with self._initialize_lock:
            assert self._initialized
            server_def_str = self._collective_ops_server_def.SerializeToString()
            pywrap_tfe.TFE_EnableCollectiveOps(self._context_handle, server_def_str)
            self._initialize_logical_devices()
            self._clear_caches()
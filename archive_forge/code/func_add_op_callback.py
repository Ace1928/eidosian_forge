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
def add_op_callback(self, callback):
    """Add a post-op callback to the context.

    A post-op callback is invoked immediately after an eager operation or
    function has finished execution or after a op has been added to a graph,
    providing access to the op's type, name input and output tensors. Multiple
    op callbacks can be added, in which case the callbacks will be invoked in
    the order in which they are added.

    Args:
      callback: a callable of the signature `f(op_type, inputs, attrs, outputs,
        op_name=None, graph=None)`. See doc strings in `op_callbacks.py` for
        details on the function signature and its semantics.
    """
    if callback not in self._thread_local_data.op_callbacks:
        self._thread_local_data.op_callbacks.append(callback)
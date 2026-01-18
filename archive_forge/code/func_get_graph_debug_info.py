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
def get_graph_debug_info(self, name):
    """Get GraphDebugInfo associated with a function from the context.

    Args:
      name: function signature name.

    Returns:
      The requested GraphDebugInfo.

    Raises:
      tf.errors.NotFoundError: if name is not the name of a registered function.
    """
    with c_api_util.tf_buffer() as buffer_:
        pywrap_tfe.TFE_ContextGetGraphDebugInfo(self._handle, name, buffer_)
        proto_data = pywrap_tf_session.TF_GetBuffer(buffer_)
    graph_debug_info = graph_debug_info_pb2.GraphDebugInfo()
    graph_debug_info.ParseFromString(proto_data)
    return graph_debug_info
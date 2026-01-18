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
@execution_mode.setter
def execution_mode(self, mode):
    """Sets execution mode for current thread."""
    if mode not in (None, SYNC, ASYNC):
        raise ValueError('Execution mode should be None/SYNC/ASYNC. Got %s' % mode)
    if mode is None:
        mode = SYNC
    enable_async = mode == ASYNC
    if self.is_async() != enable_async:
        if self._context_handle is not None:
            self.executor.wait()
            executor_new = executor.new_executor(enable_async)
            self._thread_local_data.executor = executor_new
            pywrap_tfe.TFE_ContextSetExecutorForThread(self._context_handle, executor_new.handle())
        else:
            self._default_is_async = enable_async
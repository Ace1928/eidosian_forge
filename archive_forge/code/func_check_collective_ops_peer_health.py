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
def check_collective_ops_peer_health(self, task, timeout_in_ms):
    """Check collective peer health.

    This probes each task to see if they're still alive. Note that restarted
    tasks are considered a different one, and they're considered not healthy.

    This should only be used in multi client multi worker training.

    Args:
      task: a task string, must be in the format of /job:xxx/replica:0/task:N.
      timeout_in_ms: an integer, the timeout. If zero, there's no timeout.

    Raises:
      tf.errors.UnavailableError: when a peer is down.
      tf.errors.FailedPreconditionError: when a peer is a different one from the
        one this task has talked to, e.g. the peer has restarted.
      tf.errors.InvalidArgumentError: when the task string is invalid.
    """
    self.ensure_initialized()
    pywrap_tfe.TFE_CollectiveOpsCheckPeerHealth(self._handle, task, timeout_in_ms)
import collections
import copy
import enum
import re
import sys
import threading
import types
from typing import Any, AnyStr, Callable, List, NoReturn, Pattern, Tuple, Type, Union, Optional
from absl import app
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import registry
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.framework import versions
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace as profiler_trace
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lock_util
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_stack
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import kwarg_only
from tensorflow.python.util.tf_export import tf_export
def enable_eager_execution_internal(config=None, device_policy=None, execution_mode=None, server_def=None) -> None:
    """Enables eager execution for the lifetime of this program.

  Most of the doc string for enable_eager_execution is relevant here as well.

  Args:
    config: See enable_eager_execution doc string
    device_policy: See enable_eager_execution doc string
    execution_mode: See enable_eager_execution doc string
    server_def: (Optional.) A tensorflow::ServerDef proto. Enables execution on
      remote devices. GrpcServers need to be started by creating an identical
      server_def to this, and setting the appropriate task_indexes, so that the
      servers can communicate. It will then be possible to execute operations on
      remote devices.

  Raises:
    ValueError

  """
    if config is not None and (not isinstance(config, config_pb2.ConfigProto)):
        raise TypeError('config must be a tf.ConfigProto, but got %s' % type(config))
    if device_policy not in (None, context.DEVICE_PLACEMENT_EXPLICIT, context.DEVICE_PLACEMENT_WARN, context.DEVICE_PLACEMENT_SILENT, context.DEVICE_PLACEMENT_SILENT_FOR_INT32):
        raise ValueError('device_policy must be one of None, DEVICE_PLACEMENT_*')
    if execution_mode not in (None, context.SYNC, context.ASYNC):
        raise ValueError('execution_mode must be one of None, SYNC, ASYNC')
    if context.default_execution_mode == context.GRAPH_MODE:
        graph_mode_has_been_used = _default_graph_stack._global_default_graph is not None
        if graph_mode_has_been_used:
            raise ValueError('tf.enable_eager_execution must be called at program startup.')
    context.default_execution_mode = context.EAGER_MODE
    with context._context_lock:
        if context._context is None:
            context._set_context_locked(context.Context(config=config, device_policy=device_policy, execution_mode=execution_mode, server_def=server_def))
        elif config is not None and config is not context._context._config or (device_policy is not None and device_policy is not context._context._device_policy) or (execution_mode is not None and execution_mode is not context._context._execution_mode):
            raise ValueError('Trying to change the options of an active eager execution. Context config: %s, specified config: %s. Context device policy: %s, specified device policy: %s. Context execution mode: %s,  specified execution mode %s.' % (context._context._config, config, context._context._device_policy, device_policy, context._context._execution_mode, execution_mode))
        else:
            context._context._thread_local_data.is_eager = True
    context.context = context.context_safe
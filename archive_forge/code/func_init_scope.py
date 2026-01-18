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
@tf_export('init_scope')
@tf_contextlib.contextmanager
def init_scope():
    """A context manager that lifts ops out of control-flow scopes and function-building graphs.

  There is often a need to lift variable initialization ops out of control-flow
  scopes, function-building graphs, and gradient tapes. Entering an
  `init_scope` is a mechanism for satisfying these desiderata. In particular,
  entering an `init_scope` has three effects:

    (1) All control dependencies are cleared the moment the scope is entered;
        this is equivalent to entering the context manager returned from
        `control_dependencies(None)`, which has the side-effect of exiting
        control-flow scopes like `tf.cond` and `tf.while_loop`.

    (2) All operations that are created while the scope is active are lifted
        into the lowest context on the `context_stack` that is not building a
        graph function. Here, a context is defined as either a graph or an eager
        context. Every context switch, i.e., every installation of a graph as
        the default graph and every switch into eager mode, is logged in a
        thread-local stack called `context_switches`; the log entry for a
        context switch is popped from the stack when the context is exited.
        Entering an `init_scope` is equivalent to crawling up
        `context_switches`, finding the first context that is not building a
        graph function, and entering it. A caveat is that if graph mode is
        enabled but the default graph stack is empty, then entering an
        `init_scope` will simply install a fresh graph as the default one.

    (3) The gradient tape is paused while the scope is active.

  When eager execution is enabled, code inside an init_scope block runs with
  eager execution enabled even when tracing a `tf.function`. For example:

  ```python
  tf.compat.v1.enable_eager_execution()

  @tf.function
  def func():
    # A function constructs TensorFlow graphs,
    # it does not execute eagerly.
    assert not tf.executing_eagerly()
    with tf.init_scope():
      # Initialization runs with eager execution enabled
      assert tf.executing_eagerly()
  ```

  Raises:
    RuntimeError: if graph state is incompatible with this initialization.
  """
    if context.executing_eagerly():
        with record.stop_recording():
            yield
    else:
        scope = get_default_graph().get_name_scope()
        if scope and scope[-1] != '/':
            scope = scope + '/'
        outer_context, innermost_nonempty_device_stack = _get_outer_context_and_inner_device_stack()
        outer_graph = None
        outer_device_stack = None
        try:
            with outer_context(), name_scope(scope, skip_on_eager=False), control_dependencies(None), record.stop_recording():
                context_manager = NullContextmanager
                context_manager_input = None
                if not context.executing_eagerly():
                    outer_graph = get_default_graph()
                    outer_device_stack = outer_graph._device_function_stack
                    outer_graph._device_function_stack = innermost_nonempty_device_stack
                elif innermost_nonempty_device_stack is not None:
                    for device_spec in innermost_nonempty_device_stack.peek_objs():
                        if device_spec.function is None:
                            break
                        if device_spec.raw_string:
                            context_manager = context.device
                            context_manager_input = device_spec.raw_string
                            break
                with context_manager(context_manager_input):
                    yield
        finally:
            if outer_graph is not None:
                outer_graph._device_function_stack = outer_device_stack
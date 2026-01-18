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
@tf_contextlib.contextmanager
def colocate_with(self, op, ignore_existing=False):
    """Returns a context manager that specifies an op to colocate with.

    Note: this function is not for public use, only for internal libraries.

    For example:

    ```python
    a = tf.Variable([1.0])
    with g.colocate_with(a):
      b = tf.constant(1.0)
      c = tf.add(a, b)
    ```

    `b` and `c` will always be colocated with `a`, no matter where `a`
    is eventually placed.

    **NOTE** Using a colocation scope resets any existing device constraints.

    If `op` is `None` then `ignore_existing` must be `True` and the new
    scope resets all colocation and device constraints.

    Args:
      op: The op to colocate all created ops with, or `None`.
      ignore_existing: If true, only applies colocation of this op within the
        context, rather than applying all colocation properties on the stack.
        If `op` is `None`, this value must be `True`.

    Raises:
      ValueError: if op is None but ignore_existing is False.

    Yields:
      A context manager that specifies the op with which to colocate
      newly created ops.
    """
    if op is None and (not ignore_existing):
        raise ValueError('Trying to reset colocation (op is None) but ignore_existing is not True')
    op, device_only_candidate = _op_to_colocate_with(op, self)
    device_fn_tmp = self._device_function_stack
    self._device_function_stack = traceable_stack.TraceableStack()
    if ignore_existing:
        current_stack = self._colocation_stack
        self._colocation_stack = traceable_stack.TraceableStack()
    if op is not None:
        self._colocation_stack.push_obj(op, offset=4)
        if device_only_candidate is not None:
            self._colocation_stack.push_obj(device_only_candidate, offset=4)
    elif not ignore_existing:
        raise ValueError('Trying to reset colocation (op is None) but ignore_existing is not True')
    try:
        yield
    finally:
        self._device_function_stack = device_fn_tmp
        if op is not None:
            self._colocation_stack.pop_obj()
            if device_only_candidate is not None:
                self._colocation_stack.pop_obj()
        if ignore_existing:
            self._colocation_stack = current_stack
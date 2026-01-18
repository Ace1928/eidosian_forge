import abc
import collections
import functools
import os
import re
import threading
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler as _profiler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import resource
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export('summary.record_if', v1=[])
@tf_contextlib.contextmanager
def record_if(condition):
    """Sets summary recording on or off per the provided boolean value.

  The provided value can be a python boolean, a scalar boolean Tensor, or
  or a callable providing such a value; if a callable is passed it will be
  invoked on-demand to determine whether summary writing will occur.  Note that
  when calling record_if() in an eager mode context, if you intend to provide a
  varying condition like `step % 100 == 0`, you must wrap this in a
  callable to avoid immediate eager evaluation of the condition.  In particular,
  using a callable is the only way to have your condition evaluated as part of
  the traced body of an @tf.function that is invoked from within the
  `record_if()` context.

  Args:
    condition: can be True, False, a bool Tensor, or a callable providing such.

  Yields:
    Returns a context manager that sets this value on enter and restores the
    previous value on exit.
  """
    old = _summary_state.is_recording
    try:
        _summary_state.is_recording = condition
        yield
    finally:
        _summary_state.is_recording = old
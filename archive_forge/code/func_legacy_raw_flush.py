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
def legacy_raw_flush(writer=None, name=None):
    """Legacy version of flush() that accepts a raw resource tensor for `writer`.

  Do not use this function in any new code. Not supported and not part of the
  public TF APIs.

  Args:
    writer: The `tf.summary.SummaryWriter` to flush. If None, the current
      default writer will be used instead; if there is no current writer, this
      returns `tf.no_op`. For this legacy version only, also accepts a raw
      resource tensor pointing to the underlying C++ writer resource.
    name: Ignored legacy argument for a name for the operation.

  Returns:
    The created `tf.Operation`.
  """
    if writer is None or isinstance(writer, SummaryWriter):
        return flush(writer, name)
    else:
        with ops.device('cpu:0'):
            return gen_summary_ops.flush_summary_writer(writer, name=name)
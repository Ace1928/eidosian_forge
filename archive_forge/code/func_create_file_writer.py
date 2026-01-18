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
def create_file_writer(logdir, max_queue=None, flush_millis=None, filename_suffix=None, name=None):
    """Creates a summary file writer in the current context under the given name.

  Args:
    logdir: a string, or None. If a string, creates a summary file writer
     which writes to the directory named by the string. If None, returns
     a mock object which acts like a summary writer but does nothing,
     useful to use as a context manager.
    max_queue: the largest number of summaries to keep in a queue; will
     flush once the queue gets bigger than this. Defaults to 10.
    flush_millis: the largest interval between flushes. Defaults to 120,000.
    filename_suffix: optional suffix for the event file name. Defaults to `.v2`.
    name: Shared name for this SummaryWriter resource stored to default
      Graph. Defaults to the provided logdir prefixed with `logdir:`. Note: if a
      summary writer resource with this shared name already exists, the returned
      SummaryWriter wraps that resource and the other arguments have no effect.

  Returns:
    Either a summary writer or an empty object which can be used as a
    summary writer.
  """
    if logdir is None:
        return _NoopSummaryWriter()
    logdir = str(logdir)
    with ops.device('cpu:0'):
        if max_queue is None:
            max_queue = constant_op.constant(10)
        if flush_millis is None:
            flush_millis = constant_op.constant(2 * 60 * 1000)
        if filename_suffix is None:
            filename_suffix = constant_op.constant('.v2')
        if name is None:
            name = 'logdir:' + logdir
        resource = gen_summary_ops.summary_writer(shared_name=name)
        return _LegacyResourceSummaryWriter(resource=resource, init_op_fn=functools.partial(gen_summary_ops.create_summary_file_writer, logdir=logdir, max_queue=max_queue, flush_millis=flush_millis, filename_suffix=filename_suffix))
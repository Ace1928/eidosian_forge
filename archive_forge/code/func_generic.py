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
def generic(name, tensor, metadata=None, family=None, step=None):
    """Writes a tensor summary if possible."""

    def function(tag, scope):
        if metadata is None:
            serialized_metadata = constant_op.constant('')
        elif hasattr(metadata, 'SerializeToString'):
            serialized_metadata = constant_op.constant(metadata.SerializeToString())
        else:
            serialized_metadata = metadata
        return gen_summary_ops.write_summary(_summary_state.writer._resource, _choose_step(step), array_ops.identity(tensor), tag, serialized_metadata, name=scope)
    return summary_writer_function(name, tensor, function, family=family)
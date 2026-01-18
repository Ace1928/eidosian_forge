import collections
import hashlib
import operator
import os
import os.path
import sys
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import analytics
from tensorflow.python.platform import gfile
from tensorflow.python.platform import remote_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.tpu import tensor_tracer_flags
from tensorflow.python.tpu import tensor_tracer_report
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import training_util
def op_priority(op_type):
    """Returns the priority of the op.

  If the priority of the op is k, it will be traced if trace_level>=k.
  Args:
    op_type: String name of the operation type.
  Returns:
    Integer value corresponding the priority of the op.
  """
    if op_type in ('Const', 'Shape', 'BroadcastGradientArgs', 'Range', 'VariableShape', 'Fill', 'OneHot', 'ShapeN'):
        return 7
    if op_type in ('Identity', 'Cast', 'Reshape', 'ExpandDims', 'StopGradient', 'PreventGradient', 'Squeeze', 'Gather', 'GatherNd'):
        return 6
    if op_type in ('ConcatV2', 'Concat', 'StridedSlice', 'Slice', 'Pack', 'Tile', 'CollectivePermute', 'SplitV', 'DynamicPartition'):
        return 5
    if op_type in ('Pad', 'RandomUniformInt', 'GreaterEqual'):
        return 4
    if op_type in ('Sum', 'AddV2', 'Add', 'AddN', 'BiasAdd', 'CrossReplicaSum'):
        return 3
    if op_type in ('Neg', 'Sub'):
        return 2
    if op_type in ('Mul', 'Square', 'MatMul', 'RandomUniform', 'Select', 'Maximum', 'Mean', 'Variance', 'Exp', 'Rsqrt'):
        return 1
    return 2
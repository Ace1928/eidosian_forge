import collections
import enum
from typing import Any, Callable, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
import numpy as np
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import dynamic_padding_pb2 as dynamic_padding
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as embedding_pb2
from tensorflow.python import tf2
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['tpu.batch_parallel'])
@traceback_utils.filter_traceback
def batch_parallel(computation: Callable[..., Any], inputs: Optional[List[List[Optional[core_types.Tensor]]]]=None, num_shards: int=1, infeed_queue: Optional[tpu_feed.InfeedQueue]=None, device_assignment: Optional[device_assignment_lib.DeviceAssignment]=None, name: Optional[Text]=None, xla_options: Optional[XLAOptions]=None):
    """Shards `computation` along the batch dimension for parallel execution.

  Convenience wrapper around shard().

  `inputs` must be a list of Tensors or None (equivalent to an empty list).
  Each input is split into `num_shards` pieces along the 0-th dimension, and
  computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  The outputs from all shards are concatenated back together along their 0-th
  dimension.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: A Python function that builds a computation to apply to each
      shard of the input.
    inputs: A list of input tensors or None (equivalent to an empty list). The
      0-th dimension of each Tensor must have size divisible by `num_shards`.
    num_shards: The number of shards.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to `computation`.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each shard of the computation uses
      only one core, and there is either only one shard, or the number of shards
      is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
    xla_options: An instance of `tpu.XLAOptions` which indicates the options
      passed to XLA compiler. Use `None` for default options.
  Returns:
    A list of output tensors.
  Raises:
    ValueError: If `num_shards <= 0`
  """
    return shard(computation, inputs, num_shards=num_shards, infeed_queue=infeed_queue, device_assignment=device_assignment, name=name, xla_options=xla_options)
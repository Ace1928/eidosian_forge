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
def split_compile_and_shard(computation: Callable[..., Any], inputs: Optional[List[List[Optional[core_types.Tensor]]]]=None, num_shards: int=1, input_shard_axes: Optional[List[int]]=None, outputs_from_all_shards: Union[bool, List[bool]]=True, output_shard_axes: Optional[List[int]]=None, infeed_queue: Optional[tpu_feed.InfeedQueue]=None, device_assignment: Optional[device_assignment_lib.DeviceAssignment]=None, name: Optional[Text]=None, xla_options: Optional[XLAOptions]=None) -> Tuple[ops.Operation, List[core_types.Tensor]]:
    """Shards `computation` for parallel execution.

  `inputs` must be a list of Tensors or None (equivalent to an empty list), each
  of which has a corresponding split axis (from `input_shard_axes`). Each input
  is split into `num_shards` pieces along the corresponding axis, and
  computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  If `outputs_from_all_shards` is true, the outputs from all shards of
  `computation` are concatenated back together along their `output_shard_axes`.
  Otherwise, each output is taken from an arbitrary shard.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: A Python function that builds a computation to apply to each
      shard of the input.
    inputs: A list of input tensors or None (equivalent to an empty list). Each
      input tensor has a corresponding shard axes, given by `input_shard_axes`,
      which must have size divisible by `num_shards`.
    num_shards: The number of shards.
    input_shard_axes: A list of dimensions along which to shard `inputs`, or
      `None`. `None` means "shard all inputs along dimension 0". If not `None`,
      there must be one dimension per input.
    outputs_from_all_shards: Boolean or list of boolean. For each output, if
      `True`, outputs from all shards are concatenated along the corresponding
      `output_shard_axes` entry. Otherwise, each output is taken
      from an arbitrary shard. If the argument is a boolean, the argument's
      value is used for each output.
    output_shard_axes: A list of dimensions along which to concatenate the
      outputs of `computation`, or `None`. `None` means "concatenate all outputs
      along dimension 0". If not `None`, there must be one dimension per output.
      Ignored if `outputs_from_all_shards` is False.
    infeed_queue: If not `None`, the `InfeedQueue` to use to augment the inputs
      of `computation`.
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
    A tuple of (compile op, [output tensors]).
  Raises:
    ValueError: If num_shards <= 0
    ValueError: If len(input_shard_axes) != len(inputs)
    ValueError: If len(output_shard_axes) != len(outputs from `computation`)
  """
    if num_shards <= 0:
        raise ValueError(f'num_shards must be a positive integer. Received {num_shards}')
    inputs = [] if inputs is None else inputs
    if not isinstance(inputs, list):
        raise TypeError(f"tpu.shard()'s inputs must be a list of Tensors or None. Received {type(inputs)}")
    inputs = [ops.convert_to_tensor(x) for x in inputs]
    if input_shard_axes is None:
        input_shard_axes = [0] * len(inputs)
    if len(inputs) != len(input_shard_axes):
        raise ValueError(f'Length of input_shard_axes must be equal to the number of inputs. Received {len(inputs)} inputs and {len(input_shard_axes)} input_shard_axes.')
    if inputs:
        split_inputs = [array_ops.split(x, num_shards, axis=axis) for axis, x in zip(input_shard_axes, inputs)]
        transposed_inputs = [list(i) for i in zip(*split_inputs)]
    else:
        transposed_inputs = [[]] * num_shards
    compile_op, outputs = split_compile_and_replicate(computation, transposed_inputs, infeed_queue=infeed_queue, device_assignment=device_assignment, name=name, xla_options=xla_options)
    if isinstance(outputs[0], ops.Operation):
        return (compile_op, [outputs[0]])
    num_outputs = len(outputs[0])
    if output_shard_axes is None:
        output_shard_axes = [0] * num_outputs
    if num_outputs != len(output_shard_axes):
        raise ValueError(f'Length of output_shard_axes must be equal to the number of outputs. Received {num_outputs} outputs and {len(output_shard_axes)} output_shard_axes.')
    if isinstance(outputs_from_all_shards, bool):
        outputs_from_all_shards = [outputs_from_all_shards] * num_outputs
    if num_outputs != len(outputs_from_all_shards):
        raise ValueError(f'Length of outputs_from_all_shards must be equal to the number of outputs. Received {num_outputs} outputs  and {len(outputs_from_all_shards)} outputs_from_all_shards.')
    results = []
    for axis, all_shards, x in zip(output_shard_axes, outputs_from_all_shards, zip(*outputs)):
        if all_shards:
            shape = x[0].shape
            is_scalar = shape is not None and shape.ndims == 0
            results.append(array_ops_stack.stack(list(x)) if is_scalar else array_ops.concat(list(x), axis=axis))
        else:
            results.append(x[0])
    return (compile_op, results)
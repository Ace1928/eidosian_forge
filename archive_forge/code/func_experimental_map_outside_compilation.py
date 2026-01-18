from typing import Any, Callable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def experimental_map_outside_compilation(computation: Callable[..., Any], *args, **kwargs) -> Any:
    """Maps `computation` onto shards and puts it outside any current TPU replicate scope.

  `experimental_map_outside_compilation(f, x)` maps `f` onto the shards
  of `x`, where `x` is split-sharded. Each invocation of `f` on a split occurs
  on the CPU that's associated with the TPU that owns the split.

  Example usage:

  ```python
  def normalize_each_split(split):
    return split - tf.math.reduce_mean(split)

  def tpu_computation(x):
    x_split = strategy.experimental_split_to_logical_devices(
                x, [num_cores_per_replica, 1])
    y = experimental_map_outside_compilation(
          normalize_each_split, x_split)
    y_split = strategy.experimental_split_to_logical_devices(
                x, [num_cores_per_replica, 1])
    return y_split
  ```

  `experimental_map_outside_compilation` should be called inside
  TPUReplicateContext. That is, `outside_compilation()` should be called
  inside a function that is passed to `tpu.split_compile_and_replicate()` --
  this is implied when outside compilation is invoked inside a function passed
  to TPUStrategy `run()`. It is invalid to invoke outside of
  TPUReplicateContext.

  `experimental_map_outside_compilation` should input and output tensors that
  are located on the TPU.

  Internally, `experimental_map_outside_compilation()` adds outside
  compilation attributes to all ops in `computation` and moves outside-compiled
  ops to a host-side graph. This is similar to `tf.tpu.outside_compilation()`.
  Send/recv ops from/to the TPU send each split directly to the TPU's host.

  Args:
    computation: A Python function that builds the computation to place on the
      host.
    *args: the positional arguments for the computation.
    **kwargs: the keyword arguments for the computation.

  Returns:
    The Tensors returned by computation.
  """
    return outside_compilation_impl(True, computation, *args, **kwargs)
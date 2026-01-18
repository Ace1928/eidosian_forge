import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def build_shuffle_all_reduce(input_tensors, gather_devices, red_op, un_op=None):
    """Construct a subgraph for shuffle all-reduce.

  Shuffle reduce is essentially the algorithm implemented when using
  parameter servers.  Suppose tensor length is n, there are d devices
  and g gather shards.  Each device sends a n/g length sub-tensor to
  each gather shard.  The gather shards perform a reduction across d
  fragments, then broadcast the result back to each device.  The
  devices then join the g fully reduced fragments they receive from
  the shards.  The gather shards could perform d-1 pairwise
  reductions, or one d-way reduction.  The first is better where
  reduction Op time is low compared to transmission time, the second
  better in the other case.

  Args:
    input_tensors: list of `tf.Tensor` values to be reduced.
    gather_devices: list of names of devices on which reduction shards
      should be placed.
    red_op: an n-array elementwise reduction Op
    un_op: optional elementwise unary Op to be applied to fully-reduced values.

  Returns:
    list of `tf.Tensor` which are the fully reduced tensors.
  """
    input_tensors, shape = _flatten_tensors(input_tensors)
    dst_devices = [t.device for t in input_tensors]
    reduced_shards = _build_shuffle_gather(input_tensors, gather_devices, red_op, un_op)
    output_tensors = _build_shuffle_scatter(reduced_shards, dst_devices)
    if len(shape) != 1:
        output_tensors = _reshape_tensors(output_tensors, shape)
    return output_tensors
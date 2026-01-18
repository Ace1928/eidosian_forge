import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def build_shuffle_then_ring(input_tensors, gather_devices, subdiv, red_n_op, red_op, un_op=None):
    """Construct hybrid of Shuffle within workers, Ring across workers."""

    def upper_builder(tensors):
        return build_ring_all_reduce(tensors, len(tensors), subdiv, [0], red_op, un_op)

    def upper_level_f(tensors):
        return _reduce_non_singleton(tensors, upper_builder, un_op)
    return _build_shuffle_hybrid(input_tensors, gather_devices, red_n_op, upper_level_f)
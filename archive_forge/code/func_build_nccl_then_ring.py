import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def build_nccl_then_ring(input_tensors, subdiv, red_op, un_op=None):
    """Construct hybrid of NCCL within workers, Ring across workers."""

    def upper_builder(y):
        return build_ring_all_reduce(y, len(y), subdiv, [0], red_op, un_op)

    def upper_level_f(x):
        return _reduce_non_singleton(x, upper_builder, un_op)
    return _build_nccl_hybrid(input_tensors, red_op, upper_level_f)
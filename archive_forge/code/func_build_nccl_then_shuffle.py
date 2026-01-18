import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def build_nccl_then_shuffle(input_tensors, gather_devices, nccl_red_op, shuffle_red_op, un_op=None):
    """Construct hybrid of NCCL within workers, Shuffle across workers."""

    def upper_level_f(x):
        return build_shuffle_all_reduce(x, gather_devices, shuffle_red_op, un_op)
    return _build_nccl_hybrid(input_tensors, nccl_red_op, upper_level_f)
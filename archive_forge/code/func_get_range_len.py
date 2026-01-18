from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
def get_range_len(start, limit, delta):
    dist = ops.convert_to_tensor(limit - start)
    unadjusted_len = dist // delta
    adjustment = math_ops.cast(gen_math_ops.not_equal(dist % delta, array_ops.zeros_like(unadjusted_len)), dist.dtype)
    final_len = unadjusted_len + adjustment
    return gen_math_ops.maximum(final_len, array_ops.zeros_like(final_len))
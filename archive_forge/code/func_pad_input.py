import math as _math
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def pad_input():
    rank = len(input.shape)
    padding = [[0, 0] for _ in range(rank)]
    padding[rank - 1][1] = n - seq_len
    padding = _ops.convert_to_tensor(padding, dtype=_dtypes.int32)
    return _array_ops.pad(input, paddings=padding)
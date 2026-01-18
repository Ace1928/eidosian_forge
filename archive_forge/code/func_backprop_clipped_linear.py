import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def backprop_clipped_linear(dY, X, *, slope: float=1.0, offset: float=0.0, min_val: float=0.0, max_val: float=1.0, inplace: bool=False, threads_per_block=128, num_blocks=128):
    _is_float_array(dY)
    _is_float_array(X, shape=dY.shape)
    out = dY
    if not inplace:
        out = _alloc_like(dY, zeros=False)
    if dY.dtype == 'float32':
        backprop_clipped_linear_kernel_float((num_blocks,), (threads_per_block,), (out, dY, X, slope, offset, min_val, max_val, out.size))
    else:
        backprop_clipped_linear_kernel_double((num_blocks,), (threads_per_block,), (out, dY, X, slope, offset, min_val, max_val, out.size))
    return out
import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def backprop_reduce_sum(d_sums, lengths, *, threads_per_block=128, num_blocks=128):
    _is_float_array(d_sums)
    B = len(lengths)
    T = int(lengths.sum())
    O = d_sums.shape[1]
    _check_lengths(lengths, T)
    out = _alloc((T, O), dtype=d_sums.dtype, zeros=False)
    if d_sums.dtype == 'float32':
        backprop_reduce_sum_kernel_float((num_blocks,), (threads_per_block,), (out, d_sums, lengths, B, T, O))
    else:
        backprop_reduce_sum_kernel_double((num_blocks,), (threads_per_block,), (out, d_sums, lengths, B, T, O))
    return out
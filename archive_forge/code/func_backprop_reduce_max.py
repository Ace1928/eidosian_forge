import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def backprop_reduce_max(d_maxes, which, lengths, *, threads_per_block=128, num_blocks=128):
    _is_float_array(d_maxes)
    B = len(lengths)
    T = int(lengths.sum())
    O = d_maxes.shape[1]
    _check_lengths(lengths, T, min_length=1)
    out = _alloc((T, O), dtype=d_maxes.dtype, zeros=True)
    _check_which_reduce_max(which, (B, O), lengths)
    if d_maxes.dtype == 'float32':
        backprop_reduce_max_kernel_float((num_blocks,), (threads_per_block,), (out, d_maxes, which, lengths, B, T, O))
    else:
        backprop_reduce_max_kernel_double((num_blocks,), (threads_per_block,), (out, d_maxes, which, lengths, B, T, O))
    return out
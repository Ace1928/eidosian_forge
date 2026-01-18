import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
def gen_block(builder, block_pos, block_name, bb_end, args):
    strides, _, idx, _ = args
    bb = builder.append_basic_block(name=block_name)
    with builder.goto_block(bb):
        zero = ir.IntType(64)(0)
        flat_idx = zero
        if block_pos == 0:
            for i in range(1, len_idx):
                stride = builder.extract_value(strides, i - 1)
                idx_i = builder.extract_value(idx, i)
                m = builder.mul(stride, idx_i)
                flat_idx = builder.add(flat_idx, m)
        elif 0 < block_pos < len_idx - 1:
            for i in range(0, block_pos):
                stride = builder.extract_value(strides, i)
                idx_i = builder.extract_value(idx, i)
                m = builder.mul(stride, idx_i)
                flat_idx = builder.add(flat_idx, m)
            for i in range(block_pos + 1, len_idx):
                stride = builder.extract_value(strides, i - 1)
                idx_i = builder.extract_value(idx, i)
                m = builder.mul(stride, idx_i)
                flat_idx = builder.add(flat_idx, m)
        else:
            for i in range(0, len_idx - 1):
                stride = builder.extract_value(strides, i)
                idx_i = builder.extract_value(idx, i)
                m = builder.mul(stride, idx_i)
                flat_idx = builder.add(flat_idx, m)
        builder.branch(bb_end)
    return (bb, flat_idx)
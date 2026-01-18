import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def callee_ir_validator(func_ir):
    """Checks the IR of a callee is supported for inlining
    """
    for blk in func_ir.blocks.values():
        for stmt in blk.find_insts(ir.Assign):
            if isinstance(stmt.value, ir.Yield):
                msg = 'The use of yield in a closure is unsupported.'
                raise errors.UnsupportedError(msg, loc=stmt.loc)
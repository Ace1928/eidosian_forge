import math
import operator
import sys
import numpy as np
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core.imputils import Registry, impl_ret_untracked
from numba import typeof
from numba.core import types, utils, config, cgutils
from numba.core.extending import overload
from numba.core.typing import signature
from numba.cpython.unsafe.numbers import trailing_zeros
def call_fp_intrinsic(builder, name, args):
    """
    Call a LLVM intrinsic floating-point operation.
    """
    mod = builder.module
    intr = mod.declare_intrinsic(name, [a.type for a in args])
    return builder.call(intr, args)
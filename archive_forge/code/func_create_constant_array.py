import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def create_constant_array(ty, val):
    """
    Create an LLVM-constant of a fixed-length array from Python values.

    The type provided is the type of the elements.
    """
    return ir.Constant(ir.ArrayType(ty, len(val)), val)
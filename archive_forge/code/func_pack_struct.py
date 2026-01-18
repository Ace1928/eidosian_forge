import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def pack_struct(builder, values):
    """
    Pack a sequence of values into a LLVM struct.
    """
    structty = ir.LiteralStructType([v.type for v in values])
    st = structty(ir.Undefined)
    for i, v in enumerate(values):
        st = builder.insert_value(st, v, i)
    return st
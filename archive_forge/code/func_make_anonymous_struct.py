import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def make_anonymous_struct(builder, values, struct_type=None):
    """
    Create an anonymous struct containing the given LLVM *values*.
    """
    if struct_type is None:
        struct_type = ir.LiteralStructType([v.type for v in values])
    struct_val = struct_type(ir.Undefined)
    for i, v in enumerate(values):
        struct_val = builder.insert_value(struct_val, v, i)
    return struct_val
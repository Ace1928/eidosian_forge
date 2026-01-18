import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def muladd_with_overflow(builder, a, b, c):
    """
    Compute (a * b + c) and return a (result, overflow bit) pair.
    The operands must be signed integers.
    """
    p = builder.smul_with_overflow(a, b)
    prod = builder.extract_value(p, 0)
    prod_ovf = builder.extract_value(p, 1)
    s = builder.sadd_with_overflow(prod, c)
    res = builder.extract_value(s, 0)
    ovf = builder.or_(prod_ovf, builder.extract_value(s, 1))
    return (res, ovf)
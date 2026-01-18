import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def copy_struct(dst, src, repl={}):
    """
    Copy structure from *src* to *dst* with replacement from *repl*.
    """
    repl = repl.copy()
    for k in src._datamodel._fields:
        v = repl.pop(k, getattr(src, k))
        setattr(dst, k, v)
    for k, v in repl.items():
        setattr(dst, k, v)
    return dst
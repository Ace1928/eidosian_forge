import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def memset_padding(builder, ptr):
    """
    Fill padding bytes of the pointee with zeros.
    """
    val = builder.load(ptr)
    memset(builder, ptr, sizeof(builder, ptr.type), 0)
    builder.store(val, ptr)
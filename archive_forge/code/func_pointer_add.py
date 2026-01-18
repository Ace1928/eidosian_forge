import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def pointer_add(builder, ptr, offset, return_type=None):
    """
    Add an integral *offset* to pointer *ptr*, and return a pointer
    of *return_type* (or, if omitted, the same type as *ptr*).

    Note the computation is done in bytes, and ignores the width of
    the pointed item type.
    """
    intptr = builder.ptrtoint(ptr, intp_t)
    if isinstance(offset, int):
        offset = intp_t(offset)
    intptr = builder.add(intptr, offset)
    return builder.inttoptr(intptr, return_type or ptr.type)
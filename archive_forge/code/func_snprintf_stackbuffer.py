import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def snprintf_stackbuffer(builder, bufsz, format, *args):
    """Similar to `snprintf()` but the buffer is stack allocated to size
    *bufsz*.

    Returns the buffer pointer as i8*.
    """
    assert isinstance(bufsz, int)
    spacety = ir.ArrayType(ir.IntType(8), bufsz)
    space = alloca_once(builder, spacety, zfill=True)
    buffer = builder.bitcast(space, voidptr_t)
    snprintf(builder, buffer, intp_t(bufsz), format, *args)
    return buffer
import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def guard_memory_error(context, builder, pointer, msg=None):
    """
    Guard against *pointer* being NULL (and raise a MemoryError).
    """
    assert isinstance(pointer.type, ir.PointerType), pointer.type
    exc_args = (msg,) if msg else ()
    with builder.if_then(is_null(builder, pointer), likely=False):
        context.call_conv.return_user_exc(builder, MemoryError, exc_args)
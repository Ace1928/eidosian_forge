import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def supported_ufunc_loop(ufunc, loop):
    """Return whether the *loop* for the *ufunc* is supported -in nopython-.

    *loop* should be a UFuncLoopSpec instance, and *ufunc* a numpy ufunc.

    For ufuncs implemented using the ufunc_db, it is supported if the ufunc_db
    contains a lowering definition for 'loop' in the 'ufunc' entry.

    For other ufuncs, it is type based. The loop will be considered valid if it
    only contains the following letter types: '?bBhHiIlLqQfd'. Note this is
    legacy and when implementing new ufuncs the ufunc_db should be preferred,
    as it allows for a more fine-grained incremental support.
    """
    from numba.np import ufunc_db
    loop_sig = loop.ufunc_sig
    try:
        supported_loop = loop_sig in ufunc_db.get_ufunc_info(ufunc)
    except KeyError:
        loop_types = [x.char for x in loop.numpy_inputs + loop.numpy_outputs]
        supported_types = '?bBhHiIlLqQfd'
        supported_loop = all((t in supported_types for t in loop_types))
    return supported_loop
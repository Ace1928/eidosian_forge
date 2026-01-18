import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
@contextlib.contextmanager
def register_dispatcher(disp):
    """
    Register a Dispatcher for inference while it is not yet stored
    as global or closure variable (e.g. during execution of the @jit()
    call).  This allows resolution of recursive calls with eager
    compilation.
    """
    assert callable(disp)
    assert callable(disp.py_func)
    name = disp.py_func.__name__
    _temporary_dispatcher_map[name] = disp
    _temporary_dispatcher_map_ref_count[name] += 1
    try:
        yield
    finally:
        _temporary_dispatcher_map_ref_count[name] -= 1
        if not _temporary_dispatcher_map_ref_count[name]:
            del _temporary_dispatcher_map[name]
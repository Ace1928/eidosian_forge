import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
@contextlib.contextmanager
def new_error_context(fmt_, *args, **kwargs):
    """
    A contextmanager that prepend contextual information to any exception
    raised within.  If the exception type is not an instance of NumbaError,
    it will be wrapped into a InternalError.   The exception class can be
    changed by providing a "errcls_" keyword argument with the exception
    constructor.

    The first argument is a message that describes the context.  It can be a
    format string.  If there are additional arguments, it will be used as
    ``fmt_.format(*args, **kwargs)`` to produce the final message string.
    """
    from numba.core.utils import use_old_style_errors, use_new_style_errors
    errcls = kwargs.pop('errcls_', InternalError)
    loc = kwargs.get('loc', None)
    if loc is not None and (not loc.filename.startswith(_numba_path)):
        loc_info.update(kwargs)
    try:
        yield
    except NumbaError as e:
        e.add_context(_format_msg(fmt_, args, kwargs))
        raise
    except AssertionError:
        raise
    except Exception as e:
        if use_old_style_errors():
            newerr = errcls(e).add_context(_format_msg(fmt_, args, kwargs))
            if numba.core.config.FULL_TRACEBACKS:
                tb = sys.exc_info()[2]
            else:
                tb = None
            raise newerr.with_traceback(tb)
        elif use_new_style_errors():
            raise e
        else:
            msg = f"Unknown CAPTURED_ERRORS style: '{numba.core.config.CAPTURED_ERRORS}'."
            assert 0, msg
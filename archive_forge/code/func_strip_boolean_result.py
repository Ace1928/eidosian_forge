import functools
import types
import warnings
import importlib
import sys
from gi import PyGIDeprecationWarning
from gi._gi import CallableInfo, pygobject_new_full
from gi._constants import \
from pkgutil import extend_path
def strip_boolean_result(method, exc_type=None, exc_str=None, fail_ret=None):
    """Translate method's return value for stripping off success flag.

    There are a lot of methods which return a "success" boolean and have
    several out arguments. Translate such a method to return the out arguments
    on success and None on failure.
    """

    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        ret = method(*args, **kwargs)
        if ret[0]:
            if len(ret) == 2:
                return ret[1]
            else:
                return ret[1:]
        else:
            if exc_type:
                raise exc_type(exc_str or 'call failed')
            return fail_ret
    return wrapped
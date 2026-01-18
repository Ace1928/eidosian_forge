import ctypes
import inspect
from pkg_resources import parse_version
import textwrap
import time
import types
import eventlet
from eventlet import tpool
import netaddr
from oslo_concurrency import lockutils
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
import six
from os_win import constants
from os_win import exceptions
def get_wrapped_function(function):
    """Get the method at the bottom of a stack of decorators."""
    if not hasattr(function, '__closure__') or not function.__closure__:
        return function

    def _get_wrapped_function(function):
        if not hasattr(function, '__closure__') or not function.__closure__:
            return None
        for closure in function.__closure__:
            func = closure.cell_contents
            deeper_func = _get_wrapped_function(func)
            if deeper_func:
                return deeper_func
            elif isinstance(closure.cell_contents, types.FunctionType):
                return closure.cell_contents
    return _get_wrapped_function(function)
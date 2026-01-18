from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
@builtin
def device_assert(cond, msg='', _builder=None):
    """
    Assert the condition at runtime from the device.  Requires that the environment variable :code:`TRITON_DEBUG`
    is set to a value besides :code:`0` in order for this to have any effect.

    Using the Python :code:`assert` statement is the same as calling this function, except that the second argument
    must be provided and must be a string, e.g. :code:`assert pid == 0, "pid != 0"`.  The environment variable must
    be set for this :code:`assert` statement to have any effect.

    .. highlight:: python
    .. code-block:: python

        tl.device_assert(pid == 0)
        assert pid == 0, f"pid != 0"

    :param cond: the condition to assert. This is required to be a boolean tensor.
    :param msg: the message to print if the assertion fails. This is required to be a string literal.
    """
    msg = _constexpr_to_value(msg)
    import inspect
    frame = inspect.currentframe()
    module = inspect.getmodule(frame)
    while hasattr(module, '__name__'):
        frame = frame.f_back
        module = inspect.getmodule(frame)
    lineno = 0
    func_name = 'unknown'
    file_name = 'unknown'
    if frame is not None and frame.f_back is not None:
        func_name = frame.f_code.co_name
        file_name = frame.f_back.f_code.co_filename
        lineno = frame.f_back.f_lineno
    return semantic.device_assert(_to_tensor(cond, _builder), msg, file_name, func_name, lineno, _builder)
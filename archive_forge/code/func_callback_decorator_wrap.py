import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def callback_decorator_wrap(python_callable):
    if not callable(python_callable):
        raise TypeError("the 'python_callable' argument is not callable")
    return self._backend.callback(cdecl, python_callable, error, onerror)
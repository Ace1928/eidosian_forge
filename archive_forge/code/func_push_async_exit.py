import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
def push_async_exit(self, exit):
    """Registers a coroutine function with the standard __aexit__ method
        signature.

        Can suppress exceptions the same way __aexit__ method can.
        Also accepts any object with an __aexit__ method (registering a call
        to the method instead of the object itself).
        """
    _cb_type = type(exit)
    try:
        exit_method = _cb_type.__aexit__
    except AttributeError:
        self._push_exit_callback(exit, False)
    else:
        self._push_async_cm_exit(exit, exit_method)
    return exit
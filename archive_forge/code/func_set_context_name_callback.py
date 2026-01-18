import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def set_context_name_callback(callback):
    """
    Set the callback to retrieve current context's name.

    The callback must take no arguments and return a string. For example:

    >>> import greenlet, yappi
    >>> yappi.set_context_name_callback(
    ...     lambda: greenlet.getcurrent().__class__.__name__)

    If the callback cannot return the name at this time but may be able to
    return it later, it should return None.
    """
    if callback is None:
        return _yappi.set_context_name_callback(_ctx_name_callback)
    return _yappi.set_context_name_callback(callback)
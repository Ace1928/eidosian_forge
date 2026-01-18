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
def set_context_id_callback(callback):
    """
    Use a number other than thread_id to determine the current context.

    The callback must take no arguments and return an integer. For example:

    >>> import greenlet, yappi
    >>> yappi.set_context_id_callback(lambda: id(greenlet.getcurrent()))
    """
    return _yappi.set_context_id_callback(callback)
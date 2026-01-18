import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
@contextlib.contextmanager
def backend_like(like, set_globally='auto'):
    """Context manager for setting a default backend. The argument ``like`` can
    be an explicit backend name or an ``array`` to infer it from.

    Parameters
    ----------
    like : str or array
        The backend to set. If an array, the backend of the array's class will
        be set.
    set_globally : {"auto", False, True}, optional
        Whether to set the backend globally or for the current thread:

        - True: set the backend globally.
        - False: set the backend for the current thread.
        - "auto": set the backend globally if this thread is the thread that
          imported autoray. Otherwise set the backend for the current thread.

        Only one thread should ever call this function with
        ``set_globally=True``, (by default this is importing thread).
    """
    if set_globally == 'auto':
        set_globally = threading.get_ident() == _importing_thrid
    old_backend = get_backend(get_globally=set_globally)
    try:
        set_backend(like, set_globally)
        yield
    finally:
        set_backend(old_backend, set_globally)
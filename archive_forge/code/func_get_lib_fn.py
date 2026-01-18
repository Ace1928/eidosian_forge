import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def get_lib_fn(backend, fn):
    """Cached retrieval of correct function for backend, all the logic for
    finding the correct funtion only runs the first time.

    Parameters
    ----------
    backend : str
        The module defining the array class to dispatch on.
    fn : str
        The function to retrieve.

    Returns
    -------
    callable
    """
    try:
        lib_fn = _FUNCS[backend, fn]
    except KeyError:
        lib_fn = import_lib_fn(backend, fn)
    return lib_fn
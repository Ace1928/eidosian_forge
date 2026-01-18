import functools
import itertools
import re
import sys
import warnings
from .deprecation import (
def kwarg_error(name, kw):
    """
    Generate a TypeError to be raised by function calls with wrong kwarg.

    Parameters
    ----------
    name : str
        The name of the calling function.
    kw : str or Iterable[str]
        Either the invalid keyword argument name, or an iterable yielding
        invalid keyword arguments (e.g., a ``kwargs`` dict).
    """
    if not isinstance(kw, str):
        kw = next(iter(kw))
    return TypeError(f"{name}() got an unexpected keyword argument '{kw}'")
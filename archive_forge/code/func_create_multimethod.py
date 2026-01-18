import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
from ._uarray import (  # type: ignore
def create_multimethod(*args, **kwargs):
    """
    Creates a decorator for generating multimethods.

    This function creates a decorator that can be used with an argument
    extractor in order to generate a multimethod. Other than for the
    argument extractor, all arguments are passed on to
    :obj:`generate_multimethod`.

    See Also
    --------
    generate_multimethod
        Generates a multimethod.
    """

    def wrapper(a):
        return generate_multimethod(a, *args, **kwargs)
    return wrapper
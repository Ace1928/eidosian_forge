import collections.abc
import contextlib
import contextvars
from .._utils import set_module
from .umath import (
from . import umath
@set_module('numpy')
def getbufsize():
    """
    Return the size of the buffer used in ufuncs.

    Returns
    -------
    getbufsize : int
        Size of ufunc buffer in bytes.

    """
    return umath.geterrobj()[0]
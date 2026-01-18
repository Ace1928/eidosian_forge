import types
import sys
import numbers
import functools
import copy
import inspect
def isbytes(obj):
    """
    Deprecated. Use::
        >>> isinstance(obj, bytes)
    after this import:
        >>> from future.builtins import bytes
    """
    return isinstance(obj, type(b''))
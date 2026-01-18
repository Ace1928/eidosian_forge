import types
import sys
import numbers
import functools
import copy
import inspect
def isnewbytes(obj):
    """
    Equivalent to the result of ``type(obj)  == type(newbytes)``
    in other words, it is REALLY a newbytes instance, not a Py2 native str
    object?

    Note that this does not cover subclasses of newbytes, and it is not
    equivalent to ininstance(obj, newbytes)
    """
    return type(obj).__name__ == 'newbytes'
import types
import sys
import numbers
import functools
import copy
import inspect
def istext(obj):
    """
    Deprecated. Use::
        >>> isinstance(obj, str)
    after this import:
        >>> from future.builtins import str
    """
    return isinstance(obj, type(u''))
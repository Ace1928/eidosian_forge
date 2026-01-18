import types
import sys
import numbers
import functools
import copy
import inspect
def is_new_style(cls):
    """
    Python 2.7 has both new-style and old-style classes. Old-style classes can
    be pesky in some circumstances, such as when using inheritance.  Use this
    function to test for whether a class is new-style. (Python 3 only has
    new-style classes.)
    """
    return hasattr(cls, '__class__') and ('__dict__' in dir(cls) or hasattr(cls, '__slots__'))
from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
def instanceproperty(fget=None, fset=None, fdel=None, doc=None, classval=None):
    """ Like @property, but returns ``classval`` when used as a class attribute

    >>> class MyClass(object):
    ...     '''The class docstring'''
    ...     @instanceproperty(classval=__doc__)
    ...     def __doc__(self):
    ...         return 'An object docstring'
    ...     @instanceproperty
    ...     def val(self):
    ...         return 42
    ...
    >>> MyClass.__doc__
    'The class docstring'
    >>> MyClass.val is None
    True
    >>> obj = MyClass()
    >>> obj.__doc__
    'An object docstring'
    >>> obj.val
    42
    """
    if fget is None:
        return partial(instanceproperty, fset=fset, fdel=fdel, doc=doc, classval=classval)
    return InstanceProperty(fget=fget, fset=fset, fdel=fdel, doc=doc, classval=classval)
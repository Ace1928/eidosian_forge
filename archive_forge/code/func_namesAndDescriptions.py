import sys
import weakref
from types import FunctionType
from types import MethodType
from typing import Union
from zope.interface import ro
from zope.interface._compat import _use_c_impl
from zope.interface.exceptions import Invalid
from zope.interface.ro import ro as calculate_ro
from zope.interface.declarations import implementedBy
from zope.interface.declarations import providedBy
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import InvalidInterface
from zope.interface.declarations import _empty
def namesAndDescriptions(self, all=False):
    """Return attribute names and descriptions defined by interface."""
    if not all:
        return self.__attrs.items()
    r = {}
    for base in self.__bases__[::-1]:
        r.update(dict(base.namesAndDescriptions(all)))
    r.update(self.__attrs)
    return r.items()
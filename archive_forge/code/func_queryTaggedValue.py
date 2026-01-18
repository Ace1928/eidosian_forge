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
def queryTaggedValue(self, tag, default=None):
    """
        Queries for the value associated with *tag*, returning it from the nearest
        interface in the ``__iro__``.

        If not found, returns *default*.
        """
    for iface in self.__iro__:
        value = iface.queryDirectTaggedValue(tag, _marker)
        if value is not _marker:
            return value
    return default
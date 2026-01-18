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
def changed(self, originally_changed):
    """
        We, or something we depend on, have changed.

        By the time this is called, the things we depend on,
        such as our bases, should themselves be stable.
        """
    self._v_attrs = None
    implied = self._implied
    implied.clear()
    ancestors = self._calculate_sro()
    self.__sro__ = tuple(ancestors)
    self.__iro__ = tuple([ancestor for ancestor in ancestors if isinstance(ancestor, InterfaceClass)])
    for ancestor in ancestors:
        implied[ancestor] = ()
    for dependent in tuple(self._dependents.keys() if self._dependents else ()):
        dependent.changed(originally_changed)
    self._v_attrs = None
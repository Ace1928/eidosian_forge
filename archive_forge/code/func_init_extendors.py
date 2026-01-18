import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
def init_extendors(self):
    self._extendors = {}
    for p in self._registry._provided:
        self.add_extendor(p)
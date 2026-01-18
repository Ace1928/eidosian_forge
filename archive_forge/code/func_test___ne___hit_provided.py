import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test___ne___hit_provided(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ibaz = IFoo('IBaz')
    _component = object()
    ar, _registry, _name = self._makeOne(_component)
    ar2, _, _ = self._makeOne(_component)
    ar2.provided = ibaz
    self.assertTrue(ar != ar2)
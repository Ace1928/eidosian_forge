import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_getUtilitiesFor_miss(self):
    from zope.interface.declarations import InterfaceClass

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')
    comp = self._makeOne()
    self.assertEqual(list(comp.getUtilitiesFor(ifoo)), [])
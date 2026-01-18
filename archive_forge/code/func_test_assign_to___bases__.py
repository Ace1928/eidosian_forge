import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_assign_to___bases__(self):
    base1 = self._makeOne('base1')
    base2 = self._makeOne('base2')
    comp = self._makeOne()
    comp.__bases__ = (base1, base2)
    self.assertEqual(comp.__bases__, (base1, base2))
    self.assertEqual(comp.adapters.__bases__, (base1.adapters, base2.adapters))
    self.assertEqual(comp.utilities.__bases__, (base1.utilities, base2.utilities))
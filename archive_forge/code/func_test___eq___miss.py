import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test___eq___miss(self):
    _component = object()
    _component2 = object()
    ar, _registry, _name = self._makeOne(_component)
    ar2, _, _ = self._makeOne(_component2)
    self.assertFalse(ar == ar2)
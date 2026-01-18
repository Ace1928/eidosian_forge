import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test___lt___hit(self):
    _component = object()
    ar, _registry, _name = self._makeOne(_component)
    ar2, _, _ = self._makeOne(_component)
    self.assertFalse(ar < ar2)
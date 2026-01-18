import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test___ne___hit(self):
    _component = object()
    ur, _registry, _name = self._makeOne(_component)
    ur2, _, _ = self._makeOne(_component)
    self.assertFalse(ur != ur2)
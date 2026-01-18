import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test___ne___identity(self):
    _component = object()
    ar, _registry, _name = self._makeOne(_component)
    self.assertFalse(ar != ar)
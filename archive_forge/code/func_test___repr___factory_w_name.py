import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test___repr___factory_w_name(self):

    class _Factory:
        __name__ = 'TEST'
    hr, _registry, _name = self._makeOne(_Factory())
    self.assertEqual(repr(hr), ('HandlerRegistration(_REGISTRY, [IFoo], %r, TEST, ' + "'DOCSTRING')") % _name)
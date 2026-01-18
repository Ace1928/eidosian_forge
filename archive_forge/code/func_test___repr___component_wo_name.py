import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test___repr___component_wo_name(self):

    class _Component:

        def __repr__(self):
            return 'TEST'
    _component = _Component()
    ar, _registry, _name = self._makeOne(_component)
    ar.provided = object()
    self.assertEqual(repr(ar), ('AdapterRegistration(_REGISTRY, [IBar], None, %r, TEST, ' + "'DOCSTRING')") % _name)
import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_flattened_inheritance(self):
    from zope.interface.interface import Interface
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    decl = self._makeOne(IBar)
    self.assertEqual(list(decl.flattened()), [IBar, IFoo, Interface])
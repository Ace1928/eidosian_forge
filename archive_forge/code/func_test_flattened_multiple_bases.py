import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_flattened_multiple_bases(self):
    from zope.interface.interface import Interface
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')
    decl = self._makeOne(IFoo, IBar)
    self.assertEqual(list(decl.flattened()), [IFoo, IBar, Interface])
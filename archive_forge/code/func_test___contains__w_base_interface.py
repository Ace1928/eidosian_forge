import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___contains__w_base_interface(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    decl = self._makeOne(IFoo)
    self.assertIn(IFoo, decl)
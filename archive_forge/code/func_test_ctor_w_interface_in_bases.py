import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_ctor_w_interface_in_bases(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    decl = self._makeOne(IFoo)
    self.assertEqual(list(decl.__bases__), [IFoo])
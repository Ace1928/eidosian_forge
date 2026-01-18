import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___iter___w_nested_sequence_overlap(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')
    decl = self._makeOne(IBar, (IFoo, IBar))
    self.assertEqual(list(decl), [IBar, IFoo])
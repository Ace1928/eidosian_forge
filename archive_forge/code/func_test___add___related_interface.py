import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___add___related_interface(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')
    IBaz = InterfaceClass('IBaz')
    before = self._makeOne(IFoo, IBar)
    other = self._makeOne(IBar, IBaz)
    after = before + other
    self.assertEqual(list(after), [IFoo, IBar, IBaz])
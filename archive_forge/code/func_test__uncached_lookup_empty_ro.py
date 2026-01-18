import unittest
from zope.interface.tests import OptimizationTestMixin
def test__uncached_lookup_empty_ro(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    registry = self._makeRegistry()
    alb = self._makeOne(registry)
    result = alb._uncached_lookup((IFoo,), IBar)
    self.assertEqual(result, None)
    self.assertEqual(len(alb._required), 1)
    self.assertIn(IFoo.weakref(), alb._required)
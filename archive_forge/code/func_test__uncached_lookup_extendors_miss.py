import unittest
from zope.interface.tests import OptimizationTestMixin
def test__uncached_lookup_extendors_miss(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    registry = self._makeRegistry()
    subr = self._makeSubregistry()
    subr._adapters = [{}, {}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    subr._v_lookup = alb
    result = alb._uncached_lookup((IFoo,), IBar)
    self.assertEqual(result, None)
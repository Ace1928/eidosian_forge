import unittest
from zope.interface.tests import OptimizationTestMixin
def test__uncached_lookupAll_components_miss(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    IQux = InterfaceClass('IQux')
    registry = self._makeRegistry(IFoo, IBar)
    subr = self._makeSubregistry()
    irrelevant = object()
    subr._adapters = [{}, {IFoo: {IQux: {'': irrelevant}}}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    subr._v_lookup = alb
    result = alb._uncached_lookupAll((IFoo,), IBar)
    self.assertEqual(result, ())
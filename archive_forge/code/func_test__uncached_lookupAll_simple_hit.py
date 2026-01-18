import unittest
from zope.interface.tests import OptimizationTestMixin
def test__uncached_lookupAll_simple_hit(self):
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    registry = self._makeRegistry(IFoo, IBar)
    subr = self._makeSubregistry()
    _expected = object()
    _named = object()
    subr._adapters = [{}, {IFoo: {IBar: {'': _expected, 'named': _named}}}]
    registry.ro.append(subr)
    alb = self._makeOne(registry)
    subr._v_lookup = alb
    result = alb._uncached_lookupAll((IFoo,), IBar)
    self.assertEqual(sorted(result), [('', _expected), ('named', _named)])
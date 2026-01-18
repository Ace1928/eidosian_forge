import unittest
from zope.interface.tests import OptimizationTestMixin
def test_init_extendors_after_registry_update(self):
    from zope.interface import Interface
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar', (IFoo,))
    registry = self._makeRegistry()
    alb = self._makeOne(registry)
    registry._provided = [IFoo, IBar]
    alb.init_extendors()
    self.assertEqual(sorted(alb._extendors.keys()), sorted([IBar, IFoo, Interface]))
    self.assertEqual(alb._extendors[IFoo], [IFoo, IBar])
    self.assertEqual(alb._extendors[IBar], [IBar])
    self.assertEqual(sorted(alb._extendors[Interface]), sorted([IFoo, IBar]))
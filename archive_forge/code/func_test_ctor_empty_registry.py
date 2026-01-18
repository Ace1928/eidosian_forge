import unittest
from zope.interface.tests import OptimizationTestMixin
def test_ctor_empty_registry(self):
    registry = self._makeRegistry()
    alb = self._makeOne(registry)
    self.assertEqual(alb._extendors, {})
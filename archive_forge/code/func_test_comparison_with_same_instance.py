import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_comparison_with_same_instance(self):
    iface = self._makeOne()
    self.assertFalse(iface < iface)
    self.assertTrue(iface <= iface)
    self.assertTrue(iface == iface)
    self.assertFalse(iface != iface)
    self.assertTrue(iface >= iface)
    self.assertFalse(iface > iface)
import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_comparison_with_None(self):
    iface = self._makeOne()
    self.assertTrue(iface < None)
    self.assertTrue(iface <= None)
    self.assertFalse(iface == None)
    self.assertTrue(iface != None)
    self.assertFalse(iface >= None)
    self.assertFalse(iface > None)
    self.assertFalse(None < iface)
    self.assertFalse(None <= iface)
    self.assertFalse(None == iface)
    self.assertTrue(None != iface)
    self.assertTrue(None >= iface)
    self.assertTrue(None > iface)
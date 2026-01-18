import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_isEqualOrExtendedBy_identity(self):
    iface = self._makeOne()
    self.assertTrue(iface.isEqualOrExtendedBy(iface))
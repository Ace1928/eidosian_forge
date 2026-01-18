import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getBases(self):
    iface = self._makeOne()
    sub = self._makeOne('ISub', bases=(iface,))
    self.assertEqual(sub.getBases(), (iface,))
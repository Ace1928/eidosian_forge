import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___miss(self):
    sb = self._makeOne()
    sb._implied = {}
    self.assertFalse(sb.isOrExtends(object()))
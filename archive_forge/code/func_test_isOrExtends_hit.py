import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_isOrExtends_hit(self):
    sb = self._makeOne()
    testing = object()
    sb._implied = {testing: {}}
    self.assertTrue(sb(testing))
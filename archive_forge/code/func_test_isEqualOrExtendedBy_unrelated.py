import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_isEqualOrExtendedBy_unrelated(self):
    one = self._makeOne('One')
    another = self._makeOne('Another')
    self.assertFalse(one.isEqualOrExtendedBy(another))
    self.assertFalse(another.isEqualOrExtendedBy(one))
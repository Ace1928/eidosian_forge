import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getDescriptionFor_miss(self):
    one = self._makeOne()
    self.assertRaises(KeyError, one.getDescriptionFor, 'nonesuch')
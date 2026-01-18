import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___contains___miss(self):
    one = self._makeOne()
    self.assertFalse('nonesuch' in one)
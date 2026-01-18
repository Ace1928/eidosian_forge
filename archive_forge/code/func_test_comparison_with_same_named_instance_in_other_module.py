import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_comparison_with_same_named_instance_in_other_module(self):
    one = self._makeOne('IName', __module__='zope.interface.tests.one')
    other = self._makeOne('IName', __module__='zope.interface.tests.other')
    self.assertTrue(one < other)
    self.assertFalse(other < one)
    self.assertTrue(one <= other)
    self.assertFalse(other <= one)
    self.assertFalse(one == other)
    self.assertFalse(other == one)
    self.assertTrue(one != other)
    self.assertTrue(other != one)
    self.assertFalse(one >= other)
    self.assertTrue(other >= one)
    self.assertFalse(one > other)
    self.assertTrue(other > one)
import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_dependents(self):
    empty = self._getEmpty()
    deps = empty.dependents
    self.assertEqual({}, deps)
    deps[1] = 2
    self.assertEqual({}, empty.dependents)
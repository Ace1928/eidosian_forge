import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___iter___empty(self):
    decl = self._getEmpty()
    self.assertEqual(list(decl), [])
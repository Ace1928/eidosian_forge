import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_interfaces_empty(self):
    decl = self._getEmpty()
    l = list(decl.interfaces())
    self.assertEqual(l, [])
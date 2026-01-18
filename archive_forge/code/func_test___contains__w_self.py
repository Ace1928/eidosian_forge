import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___contains__w_self(self):
    decl = self._makeOne()
    self.assertNotIn(decl, decl)
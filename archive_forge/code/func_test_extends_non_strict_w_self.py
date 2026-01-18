import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_extends_non_strict_w_self(self):
    spec = self._makeOne()
    self.assertTrue(spec.extends(spec, strict=False))
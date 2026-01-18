import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_extends_strict_w_self(self):
    spec = self._makeOne()
    self.assertFalse(spec.extends(spec, strict=True))
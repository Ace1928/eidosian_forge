import unittest
from zope.interface.tests import OptimizationTestMixin
def test_ctor_w_bases(self):
    base = self._makeOne()
    sub = self._makeOne([base])
    self.assertEqual(len(sub._v_subregistries), 0)
    self.assertEqual(len(base._v_subregistries), 1)
    self.assertIn(sub, base._v_subregistries)
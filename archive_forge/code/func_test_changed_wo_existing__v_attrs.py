import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_changed_wo_existing__v_attrs(self):
    decl = self._makeOne()
    decl.changed(decl)
    self.assertIsNone(decl._v_attrs)
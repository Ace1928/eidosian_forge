import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__empty(self):
    inst = self._makeOne(type(self), type)
    self.assertEqual(repr(inst), 'directlyProvides(TestClassProvidesRepr)')
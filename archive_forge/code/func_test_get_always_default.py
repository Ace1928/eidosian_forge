import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_get_always_default(self):
    self.assertIsNone(self._getEmpty().get('name'))
    self.assertEqual(self._getEmpty().get('name', 42), 42)
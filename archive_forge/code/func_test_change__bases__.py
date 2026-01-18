import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_change__bases__(self):
    empty = self._getEmpty()
    empty.__bases__ = ()
    self.assertEqual(self._getEmpty().__bases__, ())
    with self.assertRaises(TypeError):
        empty.__bases__ = (1,)
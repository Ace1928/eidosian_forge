import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_flattened_empty(self):
    from zope.interface.interface import Interface
    decl = self._getEmpty()
    self.assertEqual(list(decl.flattened()), [Interface])
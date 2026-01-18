import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___sro___(self):
    from zope.interface.interface import Interface
    decl = self._getEmpty()
    self.assertEqual(decl.__sro__, (decl, Interface))
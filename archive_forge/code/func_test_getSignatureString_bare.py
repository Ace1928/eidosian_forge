import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getSignatureString_bare(self):
    method = self._makeOne()
    self.assertEqual(method.getSignatureString(), '()')
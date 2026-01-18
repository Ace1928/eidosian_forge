import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getSignatureString_w_only_required(self):
    method = self._makeOne()
    method.positional = method.required = ['foo']
    self.assertEqual(method.getSignatureString(), '(foo)')
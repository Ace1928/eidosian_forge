import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getSignatureString_w_optional(self):
    method = self._makeOne()
    method.positional = method.required = ['foo']
    method.optional = {'foo': 'bar'}
    self.assertEqual(method.getSignatureString(), "(foo='bar')")
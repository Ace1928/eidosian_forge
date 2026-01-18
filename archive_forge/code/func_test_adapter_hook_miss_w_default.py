import unittest
from zope.interface.tests import OptimizationTestMixin
def test_adapter_hook_miss_w_default(self):
    req, prv, _default = (object(), object(), object())
    lb = self._makeOne()
    found = lb.adapter_hook(prv, req, '', _default)
    self.assertIs(found, _default)
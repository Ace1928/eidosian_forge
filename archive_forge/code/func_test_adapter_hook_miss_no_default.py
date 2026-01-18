import unittest
from zope.interface.tests import OptimizationTestMixin
def test_adapter_hook_miss_no_default(self):
    req, prv = (object(), object())
    lb = self._makeOne()
    found = lb.adapter_hook(prv, req, '')
    self.assertIsNone(found)
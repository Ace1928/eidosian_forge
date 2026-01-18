import unittest
from zope.interface.tests import OptimizationTestMixin
def test_lookup_w_invalid_name(self):

    def _lookup(self, required, provided, name):
        self.fail('This should never be called')
    lb = self._makeOne(uc_lookup=_lookup)
    with self.assertRaises(ValueError):
        lb.lookup(('A',), 'B', object())
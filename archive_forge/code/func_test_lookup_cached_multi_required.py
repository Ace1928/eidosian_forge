import unittest
from zope.interface.tests import OptimizationTestMixin
def test_lookup_cached_multi_required(self):
    _called_with = []
    a, b, c = (object(), object(), object())
    _results = [a, b, c]

    def _lookup(self, required, provided, name):
        _called_with.append((required, provided, name))
        return _results.pop(0)
    lb = self._makeOne(uc_lookup=_lookup)
    found = lb.lookup(('A', 'D'), 'B', 'C')
    found = lb.lookup(('A', 'D'), 'B', 'C')
    self.assertIs(found, a)
    self.assertEqual(_called_with, [(('A', 'D'), 'B', 'C')])
    self.assertEqual(_results, [b, c])
import unittest
from zope.interface.tests import OptimizationTestMixin
def test_lookupAll_cached(self):
    _called_with = []
    _results = [object(), object(), object()]

    def _lookupAll(self, required, provided):
        _called_with.append((required, provided))
        return tuple(_results)
    lb = self._makeOne(uc_lookupAll=_lookupAll)
    found = lb.lookupAll('A', 'B')
    found = lb.lookupAll('A', 'B')
    self.assertEqual(found, tuple(_results))
    self.assertEqual(_called_with, [(('A',), 'B')])
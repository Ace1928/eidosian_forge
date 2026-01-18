import unittest
from zope.interface.tests import OptimizationTestMixin
def test_lookupAll(self):
    _results_1 = [object(), object(), object()]
    _results_2 = [object(), object(), object()]
    _results = [_results_1, _results_2]

    def _lookupAll(self, required, provided):
        return tuple(_results.pop(0))
    reg = self._makeRegistry(3)
    lb = self._makeOne(reg, uc_lookupAll=_lookupAll)
    found = lb.lookupAll('A', 'B')
    self.assertEqual(found, tuple(_results_1))
    found = lb.lookupAll('A', 'B')
    self.assertEqual(found, tuple(_results_1))
    reg.ro[1]._generation += 1
    found = lb.lookupAll('A', 'B')
    self.assertEqual(found, tuple(_results_2))
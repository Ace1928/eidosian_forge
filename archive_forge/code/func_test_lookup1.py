import unittest
from zope.interface.tests import OptimizationTestMixin
def test_lookup1(self):
    _called_with = []
    a, b, c = (object(), object(), object())
    _results = [a, b, c]

    def _lookup(self, required, provided, name):
        _called_with.append((required, provided, name))
        return _results.pop(0)
    reg = self._makeRegistry(3)
    lb = self._makeOne(reg, uc_lookup=_lookup)
    found = lb.lookup1('A', 'B', 'C')
    found = lb.lookup1('A', 'B', 'C')
    self.assertIs(found, a)
    self.assertEqual(_called_with, [(('A',), 'B', 'C')])
    self.assertEqual(_results, [b, c])
    reg.ro[1]._generation += 1
    found = lb.lookup1('A', 'B', 'C')
    self.assertIs(found, b)
    self.assertEqual(_called_with, [(('A',), 'B', 'C'), (('A',), 'B', 'C')])
    self.assertEqual(_results, [c])
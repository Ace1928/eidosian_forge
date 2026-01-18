import pyomo.common.unittest as unittest
from pyomo.contrib.trustregion.filter import Filter, FilterElement
def test_FilterElement(self):
    fe = FilterElement(self.objective, self.feasible)
    self.assertEqual(fe.objective, self.objective)
    self.assertEqual(fe.feasible, self.feasible)
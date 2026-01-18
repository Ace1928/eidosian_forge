from math import fabs
from pyomo.environ import value
def check_8PP_solution(self, eight_process, results):
    self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 0.01)
    self.assertTrue(fabs(value(results.problem.upper_bound) - 68) <= 0.01)
    self.assertTrue(value(eight_process.use_unit_1or2.disjuncts[1].indicator_var))
    self.assertFalse(value(eight_process.use_unit_1or2.disjuncts[0].indicator_var))
    self.assertTrue(value(eight_process.use_unit_4or5ornot.disjuncts[0].indicator_var))
    self.assertFalse(value(eight_process.use_unit_4or5ornot.disjuncts[1].indicator_var))
    self.assertFalse(value(eight_process.use_unit_4or5ornot.disjuncts[2].indicator_var))
    self.assertTrue(value(eight_process.use_unit_6or7ornot.disjuncts[0].indicator_var))
    self.assertFalse(value(eight_process.use_unit_6or7ornot.disjuncts[1].indicator_var))
    self.assertFalse(value(eight_process.use_unit_6or7ornot.disjuncts[2].indicator_var))
    self.assertTrue(value(eight_process.use_unit_8ornot.disjuncts[0].indicator_var))
    self.assertFalse(value(eight_process.use_unit_8ornot.disjuncts[1].indicator_var))
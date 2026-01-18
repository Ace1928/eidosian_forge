import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
def test_lp(self):
    self.set_params(-1, -2, 0.1, -2)
    x, y = self.get_solution()
    opt = Gurobi()
    res = opt.solve(self.m)
    self.assertAlmostEqual(x + y, res.incumbent_objective)
    self.assertAlmostEqual(x + y, res.objective_bound)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertTrue(res.incumbent_objective is not None)
    self.assertAlmostEqual(x, self.m.x.value)
    self.assertAlmostEqual(y, self.m.y.value)
    self.set_params(-1.25, -1, 0.5, -2)
    opt.config.load_solutions = False
    res = opt.solve(self.m)
    self.assertAlmostEqual(x, self.m.x.value)
    self.assertAlmostEqual(y, self.m.y.value)
    x, y = self.get_solution()
    self.assertNotAlmostEqual(x, self.m.x.value)
    self.assertNotAlmostEqual(y, self.m.y.value)
    res.solution_loader.load_vars()
    self.assertAlmostEqual(x, self.m.x.value)
    self.assertAlmostEqual(y, self.m.y.value)
import os
from pyomo.environ import (
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
@unittest.skipIf(not cbc_available, 'The CBC solver is not available')
def test_duals_signs(self):
    m = ConcreteModel()
    m.x = Var()
    m.obj = Objective(expr=m.x)
    m.c = Constraint(expr=(-1, m.x, 1))
    m.dual = Suffix(direction=Suffix.IMPORT)
    opt = SolverFactory('cbc')
    res = opt.solve(m)
    self.assertAlmostEqual(res.problem.lower_bound, -1)
    self.assertAlmostEqual(res.problem.upper_bound, -1)
    self.assertAlmostEqual(m.dual[m.c], 1)
    m.obj.sense = maximize
    res = opt.solve(m)
    self.assertAlmostEqual(res.problem.lower_bound, 1)
    self.assertAlmostEqual(res.problem.upper_bound, 1)
    self.assertAlmostEqual(m.dual[m.c], 1)
import subprocess
import sys
import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.contrib.appsi.solvers.highs import Highs
from pyomo.contrib.appsi.base import TerminationCondition
def test_mutable_params_with_remove_cons(self):
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-10, 10))
    m.y = pe.Var()
    m.p1 = pe.Param(mutable=True)
    m.p2 = pe.Param(mutable=True)
    m.obj = pe.Objective(expr=m.y)
    m.c1 = pe.Constraint(expr=m.y >= m.x + m.p1)
    m.c2 = pe.Constraint(expr=m.y >= -m.x + m.p2)
    m.p1.value = 1
    m.p2.value = 1
    opt = Highs()
    res = opt.solve(m)
    self.assertAlmostEqual(res.best_feasible_objective, 1)
    del m.c1
    m.p2.value = 2
    res = opt.solve(m)
    self.assertAlmostEqual(res.best_feasible_objective, -8)
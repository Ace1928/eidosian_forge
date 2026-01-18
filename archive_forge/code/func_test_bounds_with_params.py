import random
import math
from typing import Type
import pyomo.environ as pe
from pyomo import gdp
from pyomo.common.dependencies import attempt_import
import pyomo.common.unittest as unittest
from pyomo.contrib.solver.results import TerminationCondition, SolutionStatus, Results
from pyomo.contrib.solver.base import SolverBase
from pyomo.contrib.solver.ipopt import Ipopt
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.core.expr.numeric_expr import LinearExpression
@parameterized.expand(input=_load_tests(all_solvers))
def test_bounds_with_params(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.y = pe.Var()
    m.p = pe.Param(mutable=True)
    m.y.setlb(m.p)
    m.p.value = 1
    m.obj = pe.Objective(expr=m.y)
    res = opt.solve(m)
    self.assertAlmostEqual(m.y.value, 1)
    m.p.value = -1
    res = opt.solve(m)
    self.assertAlmostEqual(m.y.value, -1)
    m.y.setlb(None)
    m.y.setub(m.p)
    m.obj.sense = pe.maximize
    m.p.value = 5
    res = opt.solve(m)
    self.assertAlmostEqual(m.y.value, 5)
    m.p.value = 4
    res = opt.solve(m)
    self.assertAlmostEqual(m.y.value, 4)
    m.y.setub(None)
    m.y.setlb(m.p)
    m.obj.sense = pe.minimize
    m.p.value = 3
    res = opt.solve(m)
    self.assertAlmostEqual(m.y.value, 3)
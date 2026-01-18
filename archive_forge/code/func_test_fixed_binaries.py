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
def test_fixed_binaries(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.x = pe.Var(domain=pe.Binary)
    m.y = pe.Var()
    m.obj = pe.Objective(expr=m.y)
    m.c = pe.Constraint(expr=m.y >= m.x)
    m.x.fix(0)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 0)
    m.x.fix(1)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 1)
    opt: SolverBase = opt_class()
    if opt.is_persistent():
        opt.config.auto_updates.treat_fixed_vars_as_params = False
    m.x.fix(0)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 0)
    m.x.fix(1)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 1)
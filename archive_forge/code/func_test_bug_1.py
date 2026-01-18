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
def test_bug_1(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(3, 7))
    m.y = pe.Var(bounds=(-10, 10))
    m.p = pe.Param(mutable=True, initialize=0)
    m.obj = pe.Objective(expr=m.y)
    m.c = pe.Constraint(expr=m.y >= m.p * m.x)
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(res.incumbent_objective, 0)
    m.p.value = 1
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(res.incumbent_objective, 3)
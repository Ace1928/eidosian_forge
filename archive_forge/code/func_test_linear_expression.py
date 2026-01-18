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
def test_linear_expression(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.a1 = pe.Param(mutable=True)
    m.a2 = pe.Param(mutable=True)
    m.b1 = pe.Param(mutable=True)
    m.b2 = pe.Param(mutable=True)
    m.obj = pe.Objective(expr=m.y)
    e = LinearExpression(constant=m.b1, linear_coefs=[-1, m.a1], linear_vars=[m.y, m.x])
    m.c1 = pe.Constraint(expr=e == 0)
    e = LinearExpression(constant=m.b2, linear_coefs=[-1, m.a2], linear_vars=[m.y, m.x])
    m.c2 = pe.Constraint(expr=e == 0)
    params_to_test = [(1, -1, 2, 1), (1, -2, 2, 1), (1, -1, 3, 1)]
    for a1, a2, b1, b2 in params_to_test:
        m.a1.value = a1
        m.a2.value = a2
        m.b1.value = b1
        m.b2.value = b2
        res: Results = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
        self.assertAlmostEqual(res.incumbent_objective, m.y.value)
        if res.objective_bound is None:
            bound = -math.inf
        else:
            bound = res.objective_bound
        self.assertTrue(bound <= m.y.value)
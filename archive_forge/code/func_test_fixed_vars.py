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
def test_fixed_vars(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    for treat_fixed_vars_as_params in [True, False]:
        opt: SolverBase = opt_class()
        if opt.is_persistent():
            opt.config.auto_updates.treat_fixed_vars_as_params = treat_fixed_vars_as_params
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any((name.startswith(i) for i in nl_solvers_set)):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.x.fix(0)
        m.y = pe.Var()
        a1 = 1
        a2 = -1
        b1 = 1
        b2 = 2
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= a1 * m.x + b1)
        m.c2 = pe.Constraint(expr=m.y >= a2 * m.x + b2)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 2)
        m.x.unfix()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
        self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
        m.x.fix(0)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 2)
        m.x.value = 2
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 3)
        m.x.value = 0
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 2)
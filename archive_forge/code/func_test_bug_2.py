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
def test_bug_2(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    """
        This test is for a bug where an objective containing a fixed variable does
        not get updated properly when the variable is unfixed.
        """
    for fixed_var_option in [True, False]:
        opt: SolverBase = opt_class()
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available.')
        if any((name.startswith(i) for i in nl_solvers_set)):
            if use_presolve:
                opt.config.writer_config.linear_presolve = True
            else:
                opt.config.writer_config.linear_presolve = False
        if opt.is_persistent():
            opt.config.auto_updates.treat_fixed_vars_as_params = fixed_var_option
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=3 * m.y - m.x)
        m.c = pe.Constraint(expr=m.y >= m.x)
        m.x.fix(1)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 2, 5)
        m.x.unfix()
        m.x.setlb(-9)
        m.x.setub(9)
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, -18, 5)
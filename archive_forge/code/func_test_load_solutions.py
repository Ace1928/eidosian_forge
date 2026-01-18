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
@parameterized.expand(input=all_solvers)
def test_load_solutions(self, name: str, opt_class: Type[SolverBase]):
    opt = pe.SolverFactory(name + '_v2')
    if not opt.available(exception_flag=False):
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.obj = pe.Objective(expr=m.x)
    m.c = pe.Constraint(expr=(-1, m.x, 1))
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    res = opt.solve(m, load_solutions=False)
    pe.assert_optimal_termination(res)
    self.assertIsNone(m.x.value)
    self.assertNotIn(m.c, m.dual)
    m.solutions.load_from(res)
    self.assertAlmostEqual(m.x.value, -1)
    self.assertAlmostEqual(m.dual[m.c], 1)
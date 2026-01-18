from io import StringIO
import logging
from math import fabs
from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept
import pyomo.contrib.gdpopt.tests.common_tests as ct
from pyomo.contrib.satsolver.satsolver import z3_available
from pyomo.environ import SolverFactory, value, ConcreteModel, Var, Objective, maximize
from pyomo.gdp import Disjunction
from pyomo.opt import TerminationCondition
def test_infeasible_GDP(self):
    """Test for infeasible GDP."""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 2))
    m.d = Disjunction(expr=[[m.x ** 2 >= 3, m.x >= 3], [m.x ** 2 <= -1, m.x <= -1]])
    m.o = Objective(expr=m.x)
    result = SolverFactory('gdpopt.lbb').solve(m, tee=False, minlp_solver=minlp_solver, minlp_solver_args=minlp_args)
    self.assertEqual(result.solver.termination_condition, TerminationCondition.infeasible)
    self.assertEqual(result.solver.termination_condition, TerminationCondition.infeasible)
    self.assertIsNone(m.x.value)
    self.assertIsNone(m.d.disjuncts[0].indicator_var.value)
    self.assertIsNone(m.d.disjuncts[1].indicator_var.value)
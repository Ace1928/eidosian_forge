from contextlib import redirect_stdout
from io import StringIO
import logging
from math import fabs
from os.path import join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.contrib.appsi.solvers.gurobi import Gurobi
from pyomo.contrib.gdpopt.create_oa_subproblems import (
import pyomo.contrib.gdpopt.tests.common_tests as ct
from pyomo.contrib.gdpopt.util import is_feasible, time_code
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.tests import models
from pyomo.opt import TerminationCondition
def test_force_nlp_subproblem_with_unbounded_integer_variables(self):
    m = ConcreteModel()
    m.x = Var(domain=Integers, bounds=(0, 10))
    m.y = Var(bounds=(0, 10))
    m.w = Var(domain=Integers)
    m.disjunction = Disjunction(expr=[[m.x ** 2 <= 4, m.y ** 2 <= 1], [(m.x - 1) ** 2 + (m.y - 1) ** 2 <= 4, m.y <= 4]])
    m.c = Constraint(expr=m.x + m.y == m.w)
    m.obj = Objective(expr=-m.w)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
        results = SolverFactory('gdpopt.ric').solve(m, init_algorithm='no_init', mip_solver=mip_solver, nlp_solver=nlp_solver, force_subproblem_nlp=True, iterlim=5)
    self.assertIn('No feasible solutions found.', output.getvalue().strip())
    self.assertEqual(results.solver.termination_condition, TerminationCondition.maxIterations)
    self.assertIsNone(m.x.value)
    self.assertIsNone(m.y.value)
    self.assertIsNone(m.w.value)
    self.assertIsNone(m.disjunction.disjuncts[0].indicator_var.value)
    self.assertIsNone(m.disjunction.disjuncts[1].indicator_var.value)
    self.assertIsNotNone(results.solver.user_time)
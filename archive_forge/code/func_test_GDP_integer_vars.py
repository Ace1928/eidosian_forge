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
def test_GDP_integer_vars(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))
    m.y = Var(domain=Integers, bounds=(0, 5))
    m.d = Disjunction(expr=[[m.x >= m.y, m.y >= 3.5], [m.x >= m.y, m.y >= 2.5]])
    m.o = Objective(expr=m.x)
    SolverFactory('gdpopt').solve(m, algorithm='GLOA', mip_solver=mip_solver, nlp_solver=global_nlp_solver, minlp_solver=minlp_solver)
    self.assertAlmostEqual(value(m.o.expr), 3)
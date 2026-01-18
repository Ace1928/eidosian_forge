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
def get_GDP_on_block(self):
    m = ConcreteModel()
    m.x = Var(bounds=(-5, 5))
    m.y = Var(bounds=(-2, 6))
    m.b = Block()
    m.b.disjunction = Disjunction(expr=[[m.x + m.y <= 1, m.y >= 0.5], [m.x == 2, m.y == 4], [m.x ** 2 - m.y <= 3]])
    m.disjunction = Disjunction(expr=[[m.x - m.y <= -2, m.y >= -1], [m.x == 0, m.y >= 0], [m.y ** 2 + m.x <= 3]])
    return m
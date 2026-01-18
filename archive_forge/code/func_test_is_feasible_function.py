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
def test_is_feasible_function(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 3), initialize=2)
    m.c = Constraint(expr=m.x == 2)
    GDP_LOA_Solver = SolverFactory('gdpopt.loa')
    self.assertTrue(is_feasible(m, GDP_LOA_Solver.CONFIG()))
    m.c2 = Constraint(expr=m.x <= 1)
    self.assertFalse(is_feasible(m, GDP_LOA_Solver.CONFIG()))
    m = ConcreteModel()
    m.x = Var(bounds=(0, 3), initialize=2)
    m.c = Constraint(expr=m.x >= 5)
    self.assertFalse(is_feasible(m, GDP_LOA_Solver.CONFIG()))
    m = ConcreteModel()
    m.x = Var(bounds=(3, 3), initialize=2)
    self.assertFalse(is_feasible(m, GDP_LOA_Solver.CONFIG()))
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1), initialize=2)
    self.assertFalse(is_feasible(m, GDP_LOA_Solver.CONFIG()))
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1), initialize=2)
    m.d = Disjunct()
    with self.assertRaisesRegex(NotImplementedError, 'Found active disjunct'):
        is_feasible(m, GDP_LOA_Solver.CONFIG())
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
@unittest.skipUnless(Gurobi().available(), 'APPSI Gurobi solver is not available')
def test_auto_persistent_solver(self):
    exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
    m = exfile.build_eight_process_flowsheet()
    results = SolverFactory('gdpopt.loa').solve(m, mip_solver='appsi_gurobi')
    self.assertTrue(fabs(value(m.profit.expr) - 68) <= 0.01)
    ct.check_8PP_solution(self, m, results)
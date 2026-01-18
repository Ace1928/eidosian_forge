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
def test_ignore_set_for_oa_cuts(self):
    m = self.make_convex_circle_and_circle_slice_disjunction()
    m.disjunction.disjuncts[1].GDPopt_ignore_OA = [m.disjunction.disjuncts[1].constraint[1]]
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.DEBUG):
        SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertIn('OA cut addition for disjunction_disjuncts[1].constraint[1] skipped because it is in the ignore set.', output.getvalue().strip())
    self.assertAlmostEqual(value(m.x), 0)
    self.assertAlmostEqual(value(m.y), 1)
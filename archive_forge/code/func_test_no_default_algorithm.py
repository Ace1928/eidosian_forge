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
@unittest.skipUnless(gurobi_available, 'Gurobi not available')
def test_no_default_algorithm(self):
    m = self.make_model()
    opt = SolverFactory('gdpopt')
    buf = StringIO()
    with redirect_stdout(buf):
        opt.solve(m, algorithm='RIC', tee=True, mip_solver=mip_solver, nlp_solver=nlp_solver)
    self.assertIn('using RIC algorithm', buf.getvalue())
    self.assertAlmostEqual(value(m.obj), -0.25)
    buf = StringIO()
    with redirect_stdout(buf):
        opt.solve(m, algorithm='LBB', tee=True, mip_solver=mip_solver, nlp_solver=nlp_solver, minlp_solver='gurobi')
    self.assertIn('using LBB algorithm', buf.getvalue())
    self.assertAlmostEqual(value(m.obj), -0.25)
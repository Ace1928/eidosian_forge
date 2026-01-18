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
@unittest.skipUnless(license_available, 'Problem is too big for unlicensed BARON.')
def test_LBB_strip_pack(self):
    """Test logic-based branch and bound with strip packing."""
    exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
    strip_pack = exfile.build_rect_strip_packing_model()
    SolverFactory('gdpopt.lbb').solve(strip_pack, tee=False, check_sat=True, minlp_solver=minlp_solver, minlp_solver_args=minlp_args)
    self.assertTrue(fabs(value(strip_pack.total_length.expr) - 11) <= 0.01)
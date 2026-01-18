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
@unittest.pytest.mark.expensive
def test_LBB_constrained_layout(self):
    """Test LBB with constrained layout."""
    exfile = import_file(join(exdir, 'constrained_layout', 'cons_layout_model.py'))
    cons_layout = exfile.build_constrained_layout_model()
    SolverFactory('gdpopt.lbb').solve(cons_layout, tee=False, check_sat=True, minlp_solver=minlp_solver, minlp_solver_args=minlp_args)
    objective_value = value(cons_layout.min_dist_cost.expr)
    self.assertTrue(fabs(objective_value - 41573) <= 200, 'Objective value of %s instead of 41573' % objective_value)
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
@unittest.pytest.mark.expensive
def test_RIC_constrained_layout_default_init(self):
    """Test RIC with constrained layout."""
    exfile = import_file(join(exdir, 'constrained_layout', 'cons_layout_model.py'))
    cons_layout = exfile.build_constrained_layout_model()
    SolverFactory('gdpopt.ric').solve(cons_layout, mip_solver=mip_solver, nlp_solver=nlp_solver, iterlim=120, max_slack=5)
    objective_value = value(cons_layout.min_dist_cost.expr)
    self.assertTrue(fabs(objective_value - 41573) <= 200, 'Objective value of %s instead of 41573' % objective_value)
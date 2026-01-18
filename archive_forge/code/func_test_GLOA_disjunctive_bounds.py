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
@unittest.skipUnless(license_available, 'Global NLP solver license not available.')
def test_GLOA_disjunctive_bounds(self):
    exfile = import_file(join(exdir, 'small_lit', 'nonconvex_HEN.py'))
    model = exfile.build_gdp_model()
    SolverFactory('gdpopt.gloa').solve(model, mip_solver=mip_solver, nlp_solver=global_nlp_solver, nlp_solver_args=global_nlp_solver_args, calc_disjunctive_bounds=True, tee=False)
    objective_value = value(model.objective.expr)
    self.assertAlmostEqual(objective_value * 1e-05, 1.14385, 2)
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
def test_LBB_ex_633_trespalacios(self):
    """Test LBB with Francisco thesis example."""
    exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
    model = exfile.build_simple_nonconvex_gdp()
    SolverFactory('gdpopt').solve(model, algorithm='LBB', tee=False, check_sat=True, minlp_solver=minlp_solver, minlp_solver_args=minlp_args)
    objective_value = value(model.obj.expr)
    self.assertAlmostEqual(objective_value, 4.46, 2)
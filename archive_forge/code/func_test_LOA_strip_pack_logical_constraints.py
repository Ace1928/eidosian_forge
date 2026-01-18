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
def test_LOA_strip_pack_logical_constraints(self):
    """Test logic-based outer approximation with variation of strip
        packing with some logical constraints."""
    exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
    strip_pack = exfile.build_rect_strip_packing_model()
    strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(expr=strip_pack.no_overlap[1, 3].disjuncts[2].indicator_var.lor(strip_pack.no_overlap[1, 3].disjuncts[3].indicator_var))
    strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(expr=strip_pack.no_overlap[2, 3].disjuncts[0].indicator_var.lor(strip_pack.no_overlap[2, 3].disjuncts[1].indicator_var))
    SolverFactory('gdpopt.loa').solve(strip_pack, mip_solver=mip_solver, nlp_solver=nlp_solver, subproblem_presolve=False)
    self.assertTrue(fabs(value(strip_pack.total_length.expr) - 13) <= 0.01)
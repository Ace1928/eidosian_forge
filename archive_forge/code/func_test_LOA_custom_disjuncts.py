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
def test_LOA_custom_disjuncts(self):
    """Test logic-based OA with custom disjuncts initialization."""
    exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
    eight_process = exfile.build_eight_process_flowsheet()
    initialize = [[eight_process.use_unit_1or2.disjuncts[0], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[1], eight_process.use_unit_8ornot.disjuncts[0]], [eight_process.use_unit_1or2.disjuncts[1], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[0], eight_process.use_unit_8ornot.disjuncts[0]]]

    def assert_correct_disjuncts_active(solver, subprob_util_block, discrete_problem_util_block):
        iteration = solver.initialization_iteration
        discrete_problem = discrete_problem_util_block.model()
        subprob = subprob_util_block.model()
        if iteration >= 2:
            return
        disjs_should_be_active = initialize[iteration]
        seen = set()
        for orig_disj in disjs_should_be_active:
            parent_nm = orig_disj.parent_component().name
            idx = orig_disj.index()
            discrete_problem_parent = discrete_problem.component(parent_nm)
            subprob_parent = subprob.component(parent_nm)
            self.assertIsInstance(discrete_problem_parent, Disjunct)
            self.assertIsInstance(subprob_parent, Block)
            discrete_problem_disj = discrete_problem_parent[idx]
            subprob_disj = subprob_parent[idx]
            self.assertTrue(value(discrete_problem_disj.indicator_var))
            self.assertTrue(subprob_disj.active)
            seen.add(subprob_disj)
        for disj in subprob_util_block.disjunct_list:
            if disj not in seen:
                self.assertFalse(disj.active)
    SolverFactory('gdpopt.loa').solve(eight_process, init_algorithm='custom_disjuncts', custom_init_disjuncts=initialize, mip_solver=mip_solver, nlp_solver=nlp_solver, subproblem_initialization_method=assert_correct_disjuncts_active)
    self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 0.01)
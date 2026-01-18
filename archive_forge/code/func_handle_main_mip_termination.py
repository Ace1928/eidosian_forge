import math
from io import StringIO
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn
import logging
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy import __version__
from pyomo.common.dependencies import attempt_import
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.common.collections import ComponentMap, Bunch, ComponentSet
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.mindtpy.cut_generation import add_no_good_cuts
from operator import itemgetter
from pyomo.common.errors import DeveloperError
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.opt import (
from pyomo.core import (
from pyomo.contrib.gdpopt.util import (
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.mindtpy.util import (
def handle_main_mip_termination(self, main_mip, main_mip_results):
    should_terminate = False
    if main_mip_results is not None:
        if not self.config.single_tree:
            if main_mip_results.solver.termination_condition is tc.optimal:
                self.handle_main_optimal(main_mip)
            elif main_mip_results.solver.termination_condition is tc.infeasible:
                self.handle_main_infeasible()
                self.last_iter_cuts = True
                should_terminate = True
            elif main_mip_results.solver.termination_condition is tc.unbounded:
                temp_results = self.handle_main_unbounded(main_mip)
            elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
                temp_results = self.handle_main_unbounded(main_mip)
                if temp_results.solver.termination_condition is tc.infeasible:
                    self.handle_main_infeasible()
            elif main_mip_results.solver.termination_condition is tc.maxTimeLimit:
                self.handle_main_max_timelimit(main_mip, main_mip_results)
                self.results.solver.termination_condition = tc.maxTimeLimit
            elif main_mip_results.solver.termination_condition is tc.feasible or (main_mip_results.solver.termination_condition is tc.other and main_mip_results.solution.status is SolutionStatus.feasible):
                MindtPy = main_mip.MindtPy_utils
                self.config.logger.info('MILP solver reported feasible solution, but not guaranteed to be optimal.')
                copy_var_list_values(main_mip.MindtPy_utils.variable_list, self.fixed_nlp.MindtPy_utils.variable_list, self.config, skip_fixed=False)
                self.update_suboptimal_dual_bound(main_mip_results)
                self.config.logger.info(self.log_formatter.format(self.mip_iter, 'MILP', value(MindtPy.mip_obj.expr), self.primal_bound, self.dual_bound, self.rel_gap, get_main_elapsed_time(self.timing)))
            else:
                raise ValueError('MindtPy unable to handle MILP main termination condition of %s. Solver message: %s' % (main_mip_results.solver.termination_condition, main_mip_results.solver.message))
    else:
        self.config.logger.info('Algorithm should terminate here.')
        should_terminate = True
    return should_terminate
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
def handle_main_max_timelimit(self, main_mip, main_mip_results):
    """This function handles the result of the latest iteration of solving the MIP problem
        given that solving the MIP takes too long.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        main_mip_results : [type]
            Results from solving the MIP main subproblem.
        """
    MindtPy = main_mip.MindtPy_utils
    copy_var_list_values(main_mip.MindtPy_utils.variable_list, self.fixed_nlp.MindtPy_utils.variable_list, self.config, skip_fixed=False)
    self.update_suboptimal_dual_bound(main_mip_results)
    self.config.logger.info(self.termination_condition_log_formatter.format(self.mip_iter, 'MILP', 'maxTimeLimit', self.primal_bound, self.dual_bound, self.rel_gap, get_main_elapsed_time(self.timing)))
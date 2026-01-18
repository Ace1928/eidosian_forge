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
def handle_main_optimal(self, main_mip, update_bound=True):
    """This function copies the results from 'solve_main' to the working model and updates
        the upper/lower bound. This function is called after an optimal solution is found for
        the main problem.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        update_bound : bool, optional
            Whether to update the bound, by default True.
            Bound will not be updated when handling regularization problem.
        """
    MindtPy = main_mip.MindtPy_utils
    for var in MindtPy.discrete_variable_list:
        if var.value is None:
            self.config.logger.warning(f'Integer variable {var.name} not initialized.  Setting it to its lower bound')
            var.set_value(var.lb, skip_validation=True)
    copy_var_list_values(main_mip.MindtPy_utils.variable_list, self.fixed_nlp.MindtPy_utils.variable_list, self.config, skip_fixed=False)
    if update_bound:
        self.update_dual_bound(value(MindtPy.mip_obj.expr))
        self.config.logger.info(self.log_formatter.format(self.mip_iter, 'MILP', value(MindtPy.mip_obj.expr), self.primal_bound, self.dual_bound, self.rel_gap, get_main_elapsed_time(self.timing)))
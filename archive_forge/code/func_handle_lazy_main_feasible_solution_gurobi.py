from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_no_good_cuts
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
from math import copysign
from pyomo.contrib.mindtpy.util import (
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.opt import TerminationCondition as tc
from pyomo.core import minimize, value
from pyomo.core.expr import identify_variables
def handle_lazy_main_feasible_solution_gurobi(cb_m, cb_opt, mindtpy_solver, config):
    """This function is called during the branch and bound of main MIP problem,
    more exactly when a feasible solution is found and LazyCallback is activated.

    Copy the solution to working model and update upper or lower bound.
    In LP-NLP, upper or lower bound are updated during solving the main problem.

    Parameters
    ----------
    cb_m : Pyomo model
        The MIP main problem.
    cb_opt : SolverFactory
        The gurobi_persistent solver.
    mindtpy_solver : object
        The mindtpy solver class.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    cb_opt.cbGetSolution(vars=cb_m.MindtPy_utils.variable_list)
    copy_var_list_values(cb_m.MindtPy_utils.variable_list, mindtpy_solver.fixed_nlp.MindtPy_utils.variable_list, config, skip_fixed=False)
    copy_var_list_values(cb_m.MindtPy_utils.variable_list, mindtpy_solver.mip.MindtPy_utils.variable_list, config)
    mindtpy_solver.update_dual_bound(cb_opt.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJBND))
    config.logger.info(mindtpy_solver.log_formatter.format(mindtpy_solver.mip_iter, 'restrLP', cb_opt.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJ), mindtpy_solver.primal_bound, mindtpy_solver.dual_bound, mindtpy_solver.rel_gap, get_main_elapsed_time(mindtpy_solver.timing)))
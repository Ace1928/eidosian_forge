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
def handle_lazy_main_feasible_solution(self, main_mip, mindtpy_solver, config, opt):
    """This function is called during the branch and bound of main mip, more
        exactly when a feasible solution is found and LazyCallback is activated.
        Copy the result to working model and update upper or lower bound.
        In LP-NLP, upper or lower bound are updated during solving the main problem.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        """
    self.copy_lazy_var_list_values(opt, main_mip.MindtPy_utils.variable_list, mindtpy_solver.fixed_nlp.MindtPy_utils.variable_list, config, skip_fixed=False)
    mindtpy_solver.update_dual_bound(self.get_best_objective_value())
    config.logger.info(mindtpy_solver.log_formatter.format(mindtpy_solver.mip_iter, 'restrLP', self.get_objective_value(), mindtpy_solver.primal_bound, mindtpy_solver.dual_bound, mindtpy_solver.rel_gap, get_main_elapsed_time(mindtpy_solver.timing)))
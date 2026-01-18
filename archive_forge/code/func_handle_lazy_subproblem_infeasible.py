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
def handle_lazy_subproblem_infeasible(self, fixed_nlp, mindtpy_solver, config, opt):
    """Solves feasibility NLP subproblem and adds cuts according to the specified strategy.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        """
    config.logger.info('NLP subproblem was locally infeasible.')
    mindtpy_solver.nlp_infeasible_counter += 1
    if config.calculate_dual_at_solution:
        for c in fixed_nlp.MindtPy_utils.constraint_list:
            rhs = (0 if c.upper is None else c.upper) + (0 if c.lower is None else c.lower)
            sign_adjust = 1 if c.upper is None else -1
            fixed_nlp.dual[c] = sign_adjust * max(0, sign_adjust * (rhs - value(c.body)))
        dual_values = list((fixed_nlp.dual[c] for c in fixed_nlp.MindtPy_utils.constraint_list))
    else:
        dual_values = None
    config.logger.info('Solving feasibility problem')
    feas_subproblem, feas_subproblem_results = mindtpy_solver.solve_feasibility_subproblem()
    copy_var_list_values(feas_subproblem.MindtPy_utils.variable_list, mindtpy_solver.mip.MindtPy_utils.variable_list, config)
    if config.strategy == 'OA':
        self.add_lazy_oa_cuts(mindtpy_solver.mip, dual_values, mindtpy_solver, config, opt)
        if config.add_regularization is not None:
            add_oa_cuts(mindtpy_solver.mip, dual_values, mindtpy_solver.jacobians, mindtpy_solver.objective_sense, mindtpy_solver.mip_constraint_polynomial_degree, mindtpy_solver.mip_iter, config, mindtpy_solver.timing)
    elif config.strategy == 'GOA':
        self.add_lazy_affine_cuts(mindtpy_solver, config, opt)
    if config.add_no_good_cuts:
        var_values = list((v.value for v in fixed_nlp.MindtPy_utils.variable_list))
        self.add_lazy_no_good_cuts(var_values, mindtpy_solver, config, opt)
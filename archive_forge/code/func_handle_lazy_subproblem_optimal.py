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
def handle_lazy_subproblem_optimal(self, fixed_nlp, mindtpy_solver, config, opt):
    """This function copies the optimal solution of the fixed NLP subproblem to the MIP
        main problem(explanation see below), updates bound, adds OA and no-good cuts,
        stores incumbent solution if it has been improved.

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
    if config.calculate_dual_at_solution:
        for c in fixed_nlp.tmp_duals:
            if fixed_nlp.dual.get(c, None) is None:
                fixed_nlp.dual[c] = fixed_nlp.tmp_duals[c]
            elif config.nlp_solver == 'cyipopt' and mindtpy_solver.objective_sense == minimize:
                fixed_nlp.dual[c] = -fixed_nlp.dual[c]
        dual_values = list((fixed_nlp.dual[c] for c in fixed_nlp.MindtPy_utils.constraint_list))
    else:
        dual_values = None
    main_objective = fixed_nlp.MindtPy_utils.objective_list[-1]
    mindtpy_solver.update_primal_bound(value(main_objective.expr))
    if mindtpy_solver.primal_bound_improved:
        mindtpy_solver.best_solution_found = fixed_nlp.clone()
        mindtpy_solver.best_solution_found_time = get_main_elapsed_time(mindtpy_solver.timing)
        if config.add_no_good_cuts or config.use_tabu_list:
            mindtpy_solver.stored_bound.update({mindtpy_solver.primal_bound: mindtpy_solver.dual_bound})
    config.logger.info(mindtpy_solver.fixed_nlp_log_formatter.format('*' if mindtpy_solver.primal_bound_improved else ' ', mindtpy_solver.nlp_iter, 'Fixed NLP', value(main_objective.expr), mindtpy_solver.primal_bound, mindtpy_solver.dual_bound, mindtpy_solver.rel_gap, get_main_elapsed_time(mindtpy_solver.timing)))
    copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list, mindtpy_solver.mip.MindtPy_utils.variable_list, config)
    if config.strategy == 'OA':
        self.add_lazy_oa_cuts(mindtpy_solver.mip, dual_values, mindtpy_solver, config, opt)
        if config.add_regularization is not None:
            add_oa_cuts(mindtpy_solver.mip, dual_values, mindtpy_solver.jacobians, mindtpy_solver.objective_sense, mindtpy_solver.mip_constraint_polynomial_degree, mindtpy_solver.mip_iter, config, mindtpy_solver.timing)
    elif config.strategy == 'GOA':
        self.add_lazy_affine_cuts(mindtpy_solver, config, opt)
    if config.add_no_good_cuts:
        var_values = list((v.value for v in fixed_nlp.MindtPy_utils.variable_list))
        self.add_lazy_no_good_cuts(var_values, mindtpy_solver, config, opt)
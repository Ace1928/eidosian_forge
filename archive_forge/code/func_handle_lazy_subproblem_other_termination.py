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
def handle_lazy_subproblem_other_termination(self, fixed_nlp, termination_condition, mindtpy_solver, config):
    """Handles the result of the latest iteration of solving the NLP subproblem given
        a solution that is neither optimal nor infeasible.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        termination_condition : Pyomo TerminationCondition
            The termination condition of the fixed NLP subproblem.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.

        Raises
        ------
        ValueError
            MindtPy unable to handle the termination condition of the fixed NLP subproblem.
        """
    if termination_condition is tc.maxIterations:
        config.logger.info('NLP subproblem failed to converge within iteration limit.')
        var_values = list((v.value for v in fixed_nlp.MindtPy_utils.variable_list))
    else:
        raise ValueError('MindtPy unable to handle NLP subproblem termination condition of {}'.format(termination_condition))
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
def handle_subproblem_other_termination(self, fixed_nlp, termination_condition, cb_opt=None):
    """Handles the result of the latest iteration of solving the fixed NLP subproblem given
        a solution that is neither optimal nor infeasible.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        termination_condition : Pyomo TerminationCondition
            The termination condition of the fixed NLP subproblem.
        cb_opt : SolverFactory, optional
            The gurobi_persistent solver, by default None.

        Raises
        ------
        ValueError
            MindtPy unable to handle the NLP subproblem termination condition.
        """
    if termination_condition is tc.maxIterations:
        self.config.logger.info('NLP subproblem failed to converge within iteration limit.')
        var_values = list((v.value for v in fixed_nlp.MindtPy_utils.variable_list))
        if self.config.add_no_good_cuts:
            add_no_good_cuts(self.mip, var_values, self.config, self.timing, self.mip_iter, cb_opt)
    else:
        raise ValueError('MindtPy unable to handle NLP subproblem termination condition of {}'.format(termination_condition))
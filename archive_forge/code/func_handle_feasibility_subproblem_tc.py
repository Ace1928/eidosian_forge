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
def handle_feasibility_subproblem_tc(self, subprob_terminate_cond, MindtPy):
    """Handles the result of the latest iteration of solving the feasibility NLP subproblem.

        Parameters
        ----------
        subprob_terminate_cond : Pyomo TerminationCondition
            The termination condition of the feasibility NLP subproblem.
        MindtPy : Pyomo Block
            The MindtPy_utils block.
        """
    config = self.config
    if subprob_terminate_cond in {tc.optimal, tc.locallyOptimal, tc.feasible}:
        copy_var_list_values(MindtPy.variable_list, self.working_model.MindtPy_utils.variable_list, config)
        if value(MindtPy.feas_obj.expr) <= config.zero_tolerance:
            config.logger.warning('The objective value %.4E of feasibility problem is less than zero_tolerance. This indicates that the nlp subproblem is feasible, although it is found infeasible in the previous step. Check the nlp solver output' % value(MindtPy.feas_obj.expr))
    elif subprob_terminate_cond in {tc.infeasible, tc.noSolution}:
        config.logger.error('Feasibility subproblem infeasible. This should never happen.')
        self.should_terminate = True
        self.results.solver.status = SolverStatus.error
    elif subprob_terminate_cond is tc.maxIterations:
        config.logger.error('Subsolver reached its maximum number of iterations without converging, consider increasing the iterations limit of the subsolver or reviewing your formulation.')
        self.should_terminate = True
        self.results.solver.status = SolverStatus.error
    else:
        config.logger.error('MindtPy unable to handle feasibility subproblem termination condition of {}'.format(subprob_terminate_cond))
        self.should_terminate = True
        self.results.solver.status = SolverStatus.error
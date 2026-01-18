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
def fp_loop(self):
    """Feasibility pump loop.

        This is the outermost function for the Feasibility Pump algorithm in this package; this function
        controls the progress of solving the model.

        Raises
        ------
        ValueError
            MindtPy unable to handle the termination condition of the FP-NLP subproblem.
        """
    config = self.config
    while self.fp_iter < config.fp_iteration_limit:
        with time_code(self.timing, 'fp main'):
            fp_main, fp_main_results = self.solve_fp_main()
        fp_should_terminate = self.handle_fp_main_tc(fp_main_results)
        if fp_should_terminate:
            break
        fp_nlp, fp_nlp_result = self.solve_fp_subproblem()
        if fp_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            config.logger.info(self.log_formatter.format(self.fp_iter, 'FP-NLP', value(fp_nlp.MindtPy_utils.fp_nlp_obj), self.primal_bound, self.dual_bound, self.rel_gap, get_main_elapsed_time(self.timing)))
            self.handle_fp_subproblem_optimal(fp_nlp)
        elif fp_nlp_result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
            config.logger.error('Feasibility pump NLP subproblem infeasible')
            self.should_terminate = True
            self.results.solver.status = SolverStatus.error
            return
        elif fp_nlp_result.solver.termination_condition is tc.maxIterations:
            config.logger.error('Feasibility pump NLP subproblem failed to converge within iteration limit.')
            self.should_terminate = True
            self.results.solver.status = SolverStatus.error
            return
        else:
            raise ValueError('MindtPy unable to handle NLP subproblem termination condition of {}'.format(fp_nlp_result.solver.termination_condition))
        config.call_after_subproblem_solve(fp_nlp)
        self.fp_iter += 1
    self.mip.MindtPy_utils.del_component('fp_mip_obj')
    if config.fp_main_norm == 'L1':
        self.mip.MindtPy_utils.del_component('L1_obj')
    elif config.fp_main_norm == 'L_infinity':
        self.mip.MindtPy_utils.del_component('L_infinity_obj')
    self.mip.MindtPy_utils.cuts.del_component('improving_objective_cut')
    if not config.fp_transfercuts:
        for c in self.mip.MindtPy_utils.cuts.oa_cuts:
            c.deactivate()
        for c in self.mip.MindtPy_utils.cuts.no_good_cuts:
            c.deactivate()
    if config.fp_projcuts:
        self.working_model.MindtPy_utils.cuts.del_component('fp_orthogonality_cuts')
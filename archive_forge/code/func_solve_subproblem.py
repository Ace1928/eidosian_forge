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
def solve_subproblem(self):
    """Solves the Fixed-NLP (with fixed integers).

        This function sets up the 'fixed_nlp' by fixing binaries, sets continuous variables to their initial var values,
        precomputes dual values, deactivates trivial constraints, and then solves NLP model.

        Returns
        -------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        results : SolverResults
            Results from solving the Fixed-NLP.
        """
    config = self.config
    MindtPy = self.fixed_nlp.MindtPy_utils
    self.nlp_iter += 1
    MindtPy.cuts.deactivate()
    if config.calculate_dual_at_solution:
        self.fixed_nlp.tmp_duals = ComponentMap()
        evaluation_error = False
        for c in self.fixed_nlp.MindtPy_utils.constraint_list:
            rhs = value(c.upper) if c.has_ub() else value(c.lower)
            c_geq = -1 if c.has_ub() else 1
            try:
                self.fixed_nlp.tmp_duals[c] = c_geq * max(0, c_geq * (rhs - value(c.body)))
            except (ValueError, OverflowError) as e:
                config.logger.error(e, exc_info=True)
                self.fixed_nlp.tmp_duals[c] = None
                evaluation_error = True
        if evaluation_error:
            for nlp_var, orig_val in zip(MindtPy.variable_list, self.initial_var_values):
                if not nlp_var.fixed and (not nlp_var.is_binary()):
                    nlp_var.set_value(orig_val, skip_validation=True)
    try:
        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(self.fixed_nlp, tmp=True, ignore_infeasible=False, tolerance=config.constraint_tolerance)
    except InfeasibleConstraintException as e:
        config.logger.error(e, exc_info=True)
        config.logger.error('Infeasibility detected in deactivate_trivial_constraints.')
        results = SolverResults()
        results.solver.termination_condition = tc.infeasible
        return (self.fixed_nlp, results)
    nlp_args = dict(config.nlp_solver_args)
    update_solver_timelimit(self.nlp_opt, config.nlp_solver, self.timing, config)
    with SuppressInfeasibleWarning():
        with time_code(self.timing, 'fixed subproblem'):
            results = self.nlp_opt.solve(self.fixed_nlp, tee=config.nlp_solver_tee, load_solutions=self.load_solutions, **nlp_args)
            if len(results.solution) > 0:
                self.fixed_nlp.solutions.load_from(results)
    TransformationFactory('contrib.deactivate_trivial_constraints').revert(self.fixed_nlp)
    return (self.fixed_nlp, results)
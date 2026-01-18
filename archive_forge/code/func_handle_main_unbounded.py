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
def handle_main_unbounded(self, main_mip):
    """This function handles the result of the latest iteration of solving the MIP
        problem given an unbounded solution due to the relaxation.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.

        Returns
        -------
        main_mip_results : SolverResults
            The results of the bounded main problem.
        """
    config = self.config
    MindtPy = main_mip.MindtPy_utils
    config.logger.info(self.termination_condition_log_formatter.format(self.mip_iter, 'MILP', 'Unbounded', self.primal_bound, self.dual_bound, self.rel_gap, get_main_elapsed_time(self.timing)))
    config.logger.warning('Resolving with arbitrary bound values of (-{0:.10g}, {0:.10g}) on the objective. You can change this bound with the option obj_bound.'.format(config.obj_bound))
    MindtPy.objective_bound = Constraint(expr=(-config.obj_bound, MindtPy.mip_obj.expr, config.obj_bound))
    if isinstance(self.mip_opt, PersistentSolver):
        self.mip_opt.set_instance(main_mip)
    update_solver_timelimit(self.mip_opt, config.mip_solver, self.timing, config)
    with SuppressInfeasibleWarning():
        main_mip_results = self.mip_opt.solve(main_mip, tee=config.mip_solver_tee, load_solutions=self.load_solutions, **config.mip_solver_args)
        if len(main_mip_results.solution) > 0:
            self.mip.solutions.load_from(main_mip_results)
    return main_mip_results
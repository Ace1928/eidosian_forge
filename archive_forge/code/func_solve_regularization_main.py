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
def solve_regularization_main(self):
    """This function solves the MIP main problem.

        Returns
        -------
        self.mip : Pyomo model
            The MIP stored in self.
        main_mip_results : SolverResults
            Results from solving the main MIP.
        """
    config = self.config
    self.setup_regularization_main()
    if isinstance(self.regularization_mip_opt, PersistentSolver):
        self.regularization_mip_opt.set_instance(self.mip)
    update_solver_timelimit(self.regularization_mip_opt, config.mip_regularization_solver, self.timing, config)
    main_mip_results = self.regularization_mip_opt.solve(self.mip, tee=config.mip_solver_tee, load_solutions=self.load_solutions, **dict(config.mip_solver_args))
    if len(main_mip_results.solution) > 0:
        self.mip.solutions.load_from(main_mip_results)
    if main_mip_results.solver.termination_condition is tc.optimal:
        config.logger.info(self.log_formatter.format(self.mip_iter, 'Reg ' + self.regularization_mip_type, value(self.mip.MindtPy_utils.roa_proj_mip_obj), self.primal_bound, self.dual_bound, self.rel_gap, get_main_elapsed_time(self.timing)))
    elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
        main_mip_results, _ = distinguish_mip_infeasible_or_unbounded(self.mip, config)
    self.mip.MindtPy_utils.objective_constr.deactivate()
    self.mip.MindtPy_utils.del_component('roa_proj_mip_obj')
    self.mip.MindtPy_utils.cuts.del_component('obj_reg_estimate')
    if config.add_regularization == 'level_L1':
        self.mip.MindtPy_utils.del_component('L1_obj')
    elif config.add_regularization == 'level_L_infinity':
        self.mip.MindtPy_utils.del_component('L_infinity_obj')
    return (self.mip, main_mip_results)
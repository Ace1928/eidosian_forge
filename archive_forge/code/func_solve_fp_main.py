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
def solve_fp_main(self):
    """This function solves the MIP main problem.

        Returns
        -------
        self.mip : Pyomo model
            The MIP stored in self.
        main_mip_results : SolverResults
            Results from solving the main MIP.
        """
    config = self.config
    self.setup_fp_main()
    mip_args = self.set_up_mip_solver()
    update_solver_timelimit(self.mip_opt, config.mip_solver, self.timing, config)
    main_mip_results = self.mip_opt.solve(self.mip, tee=config.mip_solver_tee, load_solutions=self.load_solutions, **mip_args)
    if len(main_mip_results.solution) > 0:
        self.mip.solutions.load_from(main_mip_results)
    if main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
        main_mip_results, _ = distinguish_mip_infeasible_or_unbounded(self.mip, config)
    return (self.mip, main_mip_results)
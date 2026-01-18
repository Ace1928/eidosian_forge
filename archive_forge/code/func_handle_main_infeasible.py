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
def handle_main_infeasible(self):
    """This function handles the result of the latest iteration of solving
        the MIP problem given an infeasible solution.
        """
    self.config.logger.info('MIP main problem is infeasible. Problem may have no more feasible binary configurations.')
    if self.mip_iter == 1:
        self.config.logger.warning('MindtPy initialization may have generated poor quality cuts.')
    self.config.logger.info('MindtPy exiting due to MILP main problem infeasibility.')
    if self.results.solver.termination_condition is None:
        if self.primal_bound == float('inf') and self.objective_sense == minimize or (self.primal_bound == float('-inf') and self.objective_sense == maximize):
            self.results.solver.termination_condition = tc.infeasible
        else:
            self.results.solver.termination_condition = tc.feasible
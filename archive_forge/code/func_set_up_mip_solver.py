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
def set_up_mip_solver(self):
    """Set up the MIP solver.

        Returns
        -------
        mainopt : SolverFactory
            The customized MIP solver.
        """
    config = self.config
    if isinstance(self.mip_opt, PersistentSolver):
        self.mip_opt.set_instance(self.mip, symbolic_solver_labels=True)
    if config.single_tree:
        self.set_up_lazy_OA_callback()
    if config.use_tabu_list:
        self.set_up_tabulist_callback()
    mip_args = dict(config.mip_solver_args)
    if config.mip_solver in {'cplex', 'cplex_persistent', 'gurobi', 'gurobi_persistent'}:
        mip_args['warmstart'] = True
    return mip_args
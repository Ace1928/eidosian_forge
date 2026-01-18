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
def set_up_lazy_OA_callback(self):
    """Sets up the lazy OA using LazyConstraintCallback.
        Currently only support CPLEX and Gurobi.
        """
    if self.config.mip_solver == 'cplex_persistent':
        lazyoa = self.mip_opt._solver_model.register_callback(single_tree.LazyOACallback_cplex)
        lazyoa.main_mip = self.mip
        lazyoa.config = self.config
        lazyoa.opt = self.mip_opt
        lazyoa.mindtpy_solver = self
        self.mip_opt._solver_model.set_warning_stream(None)
        self.mip_opt._solver_model.set_log_stream(None)
        self.mip_opt._solver_model.set_error_stream(None)
    if self.config.mip_solver == 'gurobi_persistent':
        self.mip_opt.set_callback(single_tree.LazyOACallback_gurobi)
        self.mip_opt.mindtpy_solver = self
        self.mip_opt.config = self.config
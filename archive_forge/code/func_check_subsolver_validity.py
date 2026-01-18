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
def check_subsolver_validity(self):
    """Check if the subsolvers are available and licensed."""
    if not self.mip_opt.available():
        raise ValueError(self.config.mip_solver + ' is not available.')
    if not self.mip_opt.license_is_valid():
        raise ValueError(self.config.mip_solver + ' is not licensed.')
    if not self.nlp_opt.available():
        raise ValueError(self.config.nlp_solver + ' is not available.')
    if not self.nlp_opt.license_is_valid():
        raise ValueError(self.config.nlp_solver + ' is not licensed.')
    if self.config.add_regularization is not None:
        if not self.regularization_mip_opt.available():
            raise ValueError(self.config.mip_regularization_solver + ' is not available.')
        if not self.regularization_mip_opt.license_is_valid():
            raise ValueError(self.config.mip_regularization_solver + ' is not licensed.')
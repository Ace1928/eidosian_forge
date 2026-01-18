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
def setup_fp_main(self):
    """Set up main problem for Feasibility Pump method."""
    MindtPy = self.mip.MindtPy_utils
    for c in MindtPy.constraint_list:
        if c.body.polynomial_degree() not in self.mip_constraint_polynomial_degree:
            c.deactivate()
    MindtPy.cuts.activate()
    MindtPy.del_component('mip_obj')
    MindtPy.del_component('fp_mip_obj')
    if self.config.fp_main_norm == 'L1':
        MindtPy.fp_mip_obj = generate_norm1_objective_function(self.mip, self.working_model, discrete_only=self.config.fp_discrete_only)
    elif self.config.fp_main_norm == 'L2':
        MindtPy.fp_mip_obj = generate_norm2sq_objective_function(self.mip, self.working_model, discrete_only=self.config.fp_discrete_only)
    elif self.config.fp_main_norm == 'L_infinity':
        MindtPy.fp_mip_obj = generate_norm_inf_objective_function(self.mip, self.working_model, discrete_only=self.config.fp_discrete_only)
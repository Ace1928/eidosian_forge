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
def get_solution_name_obj(self, main_mip_results):
    if self.config.mip_solver == 'cplex_persistent':
        solution_pool_names = main_mip_results._solver_model.solution.pool.get_names()
    elif self.config.mip_solver == 'gurobi_persistent':
        solution_pool_names = list(range(main_mip_results._solver_model.SolCount))
    solution_name_obj = []
    for name in solution_pool_names:
        if self.config.mip_solver == 'cplex_persistent':
            obj = main_mip_results._solver_model.solution.pool.get_objective_value(name)
        elif self.config.mip_solver == 'gurobi_persistent':
            main_mip_results._solver_model.setParam(gurobipy.GRB.Param.SolutionNumber, name)
            obj = main_mip_results._solver_model.PoolObjVal
        solution_name_obj.append([name, obj])
    solution_name_obj.sort(key=itemgetter(1), reverse=self.objective_sense == maximize)
    solution_name_obj = solution_name_obj[:self.config.num_solution_iteration]
    return solution_name_obj
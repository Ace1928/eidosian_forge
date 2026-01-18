import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
def single_use_license():
    if not gurobipy_available:
        return False
    clean_up_global_state()
    try:
        with gp.Env():
            try:
                with gp.Env():
                    return False
            except gp.GurobiError:
                return True
    except gp.GurobiError:
        return False
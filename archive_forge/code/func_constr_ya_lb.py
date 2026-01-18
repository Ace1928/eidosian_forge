import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
def constr_ya_lb(m, a):
    return m.y[a] <= 2
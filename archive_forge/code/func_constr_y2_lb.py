import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
def constr_y2_lb(m):
    return m.y[2] + m.y[5] + m.y[6] >= 2.1
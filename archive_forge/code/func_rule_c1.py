import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
import random
def rule_c1(m, i):
    return sum((model.y[j] * (random.random() - 0.5) for j in model.Y if j != i if random.randint(0, 1))) <= round(number_binary_variables / 5) * model.y[i]
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
import random
def problem_milp_unbounded():
    model = pyo.ConcreteModel('milp_unbounded')
    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
    model.y = pyo.Var(domain=pyo.Binary)
    model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2] + model.y, sense=pyo.maximize)
    model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)
    return model
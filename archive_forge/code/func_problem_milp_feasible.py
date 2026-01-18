import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
import random
def problem_milp_feasible():
    model = pyo.ConcreteModel('milp_feasible')
    random.seed(6254)
    number_binary_variables = 20
    model.Y = pyo.RangeSet(number_binary_variables)
    model.y = pyo.Var(model.Y, domain=pyo.Binary)
    model.OBJ = pyo.Objective(expr=sum((model.y[j] * random.random() for j in model.Y)), sense=pyo.maximize)
    model.Constraint1 = pyo.Constraint(expr=sum((model.y[j] * random.random() for j in model.Y)) <= round(number_binary_variables / 5))

    def rule_c1(m, i):
        return sum((model.y[j] * (random.random() - 0.5) for j in model.Y if j != i if random.randint(0, 1))) <= round(number_binary_variables / 5) * model.y[i]
    model.constr_c1 = pyo.Constraint(model.Y, rule=rule_c1)
    return model
from pyomo.environ import (
def constr2_rule(model):
    return model.x / model.a >= model.y
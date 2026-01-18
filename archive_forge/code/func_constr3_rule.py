from pyomo.environ import (
def constr3_rule(model):
    return model.z <= model.y + model.a
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.symbol_map import symbol_map_from_instance
def c2_(model, i):
    if i == 1:
        return model.x <= 2
    elif i == 2:
        return (3, model.x, 4)
    else:
        return model.x == 5
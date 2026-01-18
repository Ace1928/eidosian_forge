import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
def start_rule(m, i):
    return (1 - i, 13 + i)
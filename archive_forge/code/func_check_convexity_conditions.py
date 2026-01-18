from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
def check_convexity_conditions(self, relax=False):
    """Returns True if all convexity conditions for the
        conic constraint are satisfied. If relax is True,
        then variable domains are ignored and it is assumed
        that all variables are continuous."""
    alpha = value(self.alpha, exception=False)
    return (relax or (self.r1.is_continuous() and self.r2.is_continuous() and all((xi.is_continuous() for xi in self.x)))) and (self.r1.has_lb() and value(self.r1.lb) >= 0) and (self.r2.has_lb() and value(self.r2.lb) >= 0) and (alpha is not None and 0 < alpha < 1)
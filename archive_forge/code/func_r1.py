from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
@property
def r1(self):
    return self._r1
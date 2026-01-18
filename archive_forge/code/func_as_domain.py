from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
@classmethod
def as_domain(cls, x):
    """Builds a conic domain. Input arguments take the
        same form as those of the conic constraint, but in
        place of each variable, one can optionally supply a
        constant, linear expression, or None.

        Returns
        -------
        block
            A block object with the core conic constraint
            (block.q) expressed using auxiliary variables
            (block.r, block.x) linked to the input arguments
            through auxiliary constraints (block.c)."""
    b = block()
    b.x = variable_tuple([variable() for i in range(len(x))])
    b.c = _build_linking_constraints(list(x), list(b.x))
    b.q = cls(x=b.x)
    return b
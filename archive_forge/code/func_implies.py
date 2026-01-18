import sys
import logging
from pyomo.common.deprecation import deprecated
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.expr.expr_common import _and, _or, _equiv, _inv, _xor, _impl
from pyomo.core.pyomoobject import PyomoObject
def implies(self, other):
    """
        Construct an ImplicationExpression using method "implies"
        """
    ans = _generate_logical_proposition(_impl, self, other)
    if ans is NotImplemented:
        raise TypeError(f"unsupported operand type for implies(): '{type(other).__name__}'")
    return ans
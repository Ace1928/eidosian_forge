import sys
import logging
from pyomo.common.deprecation import deprecated
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.expr.expr_common import _and, _or, _equiv, _inv, _xor, _impl
from pyomo.core.pyomoobject import PyomoObject
def lor(self, other):
    """
        Construct an OrExpression (Logical OR) between this BooleanValue and `other`.
        """
    ans = _generate_logical_proposition(_or, self, other)
    if ans is NotImplemented:
        raise TypeError(f"unsupported operand type for lor(): '{type(other).__name__}'")
    return ans
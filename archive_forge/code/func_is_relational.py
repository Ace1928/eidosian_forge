import sys
import logging
from pyomo.common.deprecation import deprecated
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.expr.expr_common import _and, _or, _equiv, _inv, _xor, _impl
from pyomo.core.pyomoobject import PyomoObject
@deprecated('is_relational() is deprecated in favor of is_expression_type(ExpressionType.RELATIONAL)', version='6.4.3')
def is_relational(self):
    """
        Return True if this Logical value represents a relational expression.
        """
    return False
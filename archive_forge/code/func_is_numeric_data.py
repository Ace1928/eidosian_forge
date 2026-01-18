import sys
import logging
from pyomo.common.deprecation import (
from pyomo.core.expr.expr_common import ExpressionType
from pyomo.core.expr.numeric_expr import NumericValue
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
def is_numeric_data(obj):
    """
    A utility function that returns a boolean indicating
    whether the input object is numeric and not potentially
    variable.
    """
    if obj.__class__ in native_numeric_types:
        return True
    elif obj.__class__ in native_types:
        return False
    try:
        return not obj.is_potentially_variable()
    except AttributeError:
        pass
    return check_if_numeric_type(obj)
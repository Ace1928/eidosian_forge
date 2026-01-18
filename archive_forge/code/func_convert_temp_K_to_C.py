import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def convert_temp_K_to_C(self, value_in_K):
    """
        Convert a value in Kelvin to degrees Celsius.  Note that this method
        converts a numerical value only. If you need temperature
        conversions in expressions, please work in absolute
        temperatures only.
        """
    return self._pint_convert_temp_from_to(value_in_K, self._pint_registry.K, self._pint_registry.degC)
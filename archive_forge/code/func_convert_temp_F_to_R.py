import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def convert_temp_F_to_R(self, value_in_F):
    """
        Convert a value in degrees Fahrenheit to Rankine.  Note that
        this method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
    return self._pint_convert_temp_from_to(value_in_F, self._pint_registry.degF, self._pint_registry.rankine)
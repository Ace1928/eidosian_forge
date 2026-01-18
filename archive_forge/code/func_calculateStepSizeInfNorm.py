import logging
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.trustregion.util import minIgnoreNone, maxIgnoreNone
from pyomo.core import (
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.visitor import identify_variables, ExpressionReplacementVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
from pyomo.opt import SolverFactory, check_optimal_termination
def calculateStepSizeInfNorm(self, original_values, new_values):
    """
        Taking original and new values, calculate the step-size norm ||s_k||:
            || u - u_k ||_inf

        We assume that the user has correctly scaled their variables.
        """
    original_vals = []
    new_vals = []
    for var, val in original_values.items():
        original_vals.append(val)
        new_vals.append(new_values[var])
    return max([abs(new - old) for new, old in zip(new_vals, original_vals)])
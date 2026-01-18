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
def replaceEF(self, expr):
    """
        Replace an External Function.

        Arguments:
            expr  : a Pyomo expression. We will search this expression tree

        This function returns an expression after removing any
        ExternalFunction in the set efSet from the expression tree
        `expr` and replacing them with variables.
        New variables are declared on the `TRF` block.

        TODO: Future work - investigate direct substitution of basis or
        surrogate models using Expression objects instead of new variables.
        """
    return EFReplacement(self.data, self.efSet).walk_expression(expr)
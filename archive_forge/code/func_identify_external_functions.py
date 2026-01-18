from pyomo.core.base.block import Block
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.expression import Expression
from pyomo.core.base.external import ExternalFunction
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
def identify_external_functions(expr):
    yield from _ExternalFunctionVisitor().walk_expression(expr)
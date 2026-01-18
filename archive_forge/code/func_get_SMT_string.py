import math
from pyomo.common.dependencies import attempt_import
from pyomo.core import value, SymbolMap, NumericLabeler, Var, Constraint
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.gdp import Disjunction
def get_SMT_string(self):
    prefix_string = ''.join(self.prefix_expr_list)
    variable_string = ''.join(self.variable_list)
    bounds_string = ''.join(self.bounds_list)
    expression_string = ''.join(self.expression_list)
    disjunctions_string = ''.join([self._compute_disjunction_string(d) for d in self.disjunctions_list])
    smtstring = prefix_string + variable_string + bounds_string + expression_string + disjunctions_string
    return smtstring
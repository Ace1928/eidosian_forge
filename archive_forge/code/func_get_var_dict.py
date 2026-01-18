import math
from pyomo.common.dependencies import attempt_import
from pyomo.core import value, SymbolMap, NumericLabeler, Var, Constraint
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.gdp import Disjunction
def get_var_dict(self):
    labels = [x for x in self.variable_label_map.bySymbol]
    labels.sort()
    vars = [self.variable_label_map.getObject(l) for l in labels]
    return zip(labels, vars)
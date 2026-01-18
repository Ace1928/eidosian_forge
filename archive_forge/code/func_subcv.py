import ctypes
import logging
import os
from pyomo.common.fileutils import Library
from pyomo.core import value, Expression
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.common.collections import ComponentMap
def subcv(self):
    self.warn_if_var_missing_value()
    ans = ComponentMap()
    for key in self.visitor.var_to_idx:
        i = self.visitor.var_to_idx[key]
        ans[key] = self.mcpp.subcv(self.mc_expr, i)
    return ans
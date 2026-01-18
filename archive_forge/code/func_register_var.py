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
def register_var(self, var, lb, ub):
    """Registers a new variable."""
    var_idx = self.var_to_idx[var]
    inf = float('inf')
    lb = -inf if lb is None else lb
    ub = inf if ub is None else ub
    lb = max(var.lb if var.has_lb() else -inf, lb)
    ub = min(var.ub if var.has_ub() else inf, ub)
    var_val = value(var, exception=False)
    if lb == -inf:
        lb = -500000
        logger.warning('Var %s missing lower bound. Assuming LB of %s' % (var.name, lb))
    if ub == inf:
        ub = 500000
        logger.warning('Var %s missing upper bound. Assuming UB of %s' % (var.name, ub))
    if var_val is None:
        var_val = (lb + ub) / 2
        self.missing_value_warnings.append('Var %s missing value. Assuming midpoint value of %s' % (var.name, var_val))
    return self.mcpp.newVar(lb, var_val, ub, self.num_vars, var_idx)
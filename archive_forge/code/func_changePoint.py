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
def changePoint(self, var, point):
    var.set_value(point)
    self.visitor = MCPP_visitor(self.visitor.expr)
    self.mcpp.release(self.mc_expr)
    self.mc_expr = self.visitor.walk_expression()
import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
def evaluate_abex(self):

    def before(node, child, idx):
        if type(child) in native_types or not child.is_expression_type():
            return (False, value(child))
        return (True, None)

    def enter(node):
        return (None, 0)

    def accept(node, data, child_result, child_idx):
        return data + child_result

    def exit(node, data):
        return data + 1
    return StreamBasedExpressionVisitor(beforeChild=before, acceptChildResult=accept, enterNode=enter, exitNode=exit)
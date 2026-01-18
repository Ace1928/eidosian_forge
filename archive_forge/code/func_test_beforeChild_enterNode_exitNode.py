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
def test_beforeChild_enterNode_exitNode(self):
    i = [0]

    def before(node, child, child_idx):
        if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
            return (False, [child])

    def enter(node):
        i[0] += 1
        return (None, [i[0]])

    def exit(node, data):
        if hasattr(node, 'getname'):
            data.insert(0, node.getname())
        else:
            data.insert(0, str(node))
        return data
    walker = StreamBasedExpressionVisitor(beforeChild=before, enterNode=enter, exitNode=exit)
    ans = self.walk(walker, self.e)
    m = self.m
    ref = ['sum', 1, ['pow', 2, [m.x], [2]], [m.y], ['prod', 3, [m.z], ['sum', 4, [m.x], [m.y]]]]
    self.assertEqual(str(ans), str(ref))
    ans = self.walk(walker, m.x)
    ref = ['x', 5]
    self.assertEqual(str(ans), str(ref))
    ans = self.walk(walker, 2)
    ref = ['2', 6]
    self.assertEqual(str(ans), str(ref))
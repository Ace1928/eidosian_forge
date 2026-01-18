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
def test_OLD_beforeChild_acceptChildResult_afterChild(self):
    counts = [0, 0, 0]

    def before(node, child):
        counts[0] += 1
        if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
            return (False, None)

    def accept(node, data, child_result):
        counts[1] += 1

    def after(node, child):
        counts[2] += 1
    os = StringIO()
    with LoggingIntercept(os, 'pyomo'):
        walker = StreamBasedExpressionVisitor(beforeChild=before, acceptChildResult=accept, afterChild=after)
    self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the beforeChild() method', os.getvalue().replace('\n', ' '))
    self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the acceptChildResult() method', os.getvalue().replace('\n', ' '))
    self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the afterChild() method', os.getvalue().replace('\n', ' '))
    ans = self.walk(walker, self.e)
    m = self.m
    self.assertEqual(ans, None)
    self.assertEqual(counts, [9, 9, 9])
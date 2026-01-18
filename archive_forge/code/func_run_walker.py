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
def run_walker(self, walker):
    m = self.m
    m.x = 10
    self.assertEqual(2 * RECURSION_LIMIT + 10, walker.walk_expression(m.e[2 * RECURSION_LIMIT - 1]))
    self.assertEqual(2 * RECURSION_LIMIT + 10, walker.walk_expression_nonrecursive(m.e[2 * RECURSION_LIMIT - 1]))
    TESTING_OVERHEAD = 14
    warn_msg = 'Unexpected RecursionError walking an expression tree.\n'
    if platform.python_implementation() == 'PyPy':
        cases = [(0, '')]
    elif os.environ.get('GITHUB_ACTIONS', '') and sys.platform.startswith('win'):
        cases = []
    else:
        cases = [(0, ''), (10, warn_msg)]
    head_room = sys.getrecursionlimit() - get_stack_depth()
    for n, msg in cases:
        with LoggingIntercept() as LOG:
            self.assertEqual(2 * RECURSION_LIMIT + 10, fill_stack(head_room - RECURSION_LIMIT - TESTING_OVERHEAD + n, walker.walk_expression, m.e[2 * RECURSION_LIMIT - 1]))
        self.assertEqual(msg, LOG.getvalue())
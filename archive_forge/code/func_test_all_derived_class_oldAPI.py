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
def test_all_derived_class_oldAPI(self):

    def name(x):
        if type(x) in nonpyomo_leaf_types:
            return str(x)
        else:
            return x.name

    class all_callbacks(StreamBasedExpressionVisitor):

        def __init__(self):
            self.ans = []
            super(all_callbacks, self).__init__()

        def enterNode(self, node):
            self.ans.append('Enter %s' % name(node))

        def exitNode(self, node, data):
            self.ans.append('Exit %s' % name(node))

        def beforeChild(self, node, child):
            self.ans.append('Before %s (from %s)' % (name(child), name(node)))

        def acceptChildResult(self, node, data, child_result):
            self.ans.append('Accept into %s' % name(node))

        def afterChild(self, node, child):
            self.ans.append('After %s (from %s)' % (name(child), name(node)))

        def finalizeResult(self, result):
            self.ans.append('Finalize')
    os = StringIO()
    with LoggingIntercept(os, 'pyomo'):
        walker = all_callbacks()
    self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the beforeChild() method', os.getvalue().replace('\n', ' '))
    self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the acceptChildResult() method', os.getvalue().replace('\n', ' '))
    self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the afterChild() method', os.getvalue().replace('\n', ' '))
    self.assertIsNone(self.walk(walker, self.e))
    self.assertEqual('\n'.join(walker.ans), 'Enter sum\nBefore pow (from sum)\nEnter pow\nBefore x (from pow)\nEnter x\nExit x\nAccept into pow\nAfter x (from pow)\nBefore 2 (from pow)\nEnter 2\nExit 2\nAccept into pow\nAfter 2 (from pow)\nExit pow\nAccept into sum\nAfter pow (from sum)\nBefore y (from sum)\nEnter y\nExit y\nAccept into sum\nAfter y (from sum)\nBefore prod (from sum)\nEnter prod\nBefore z (from prod)\nEnter z\nExit z\nAccept into prod\nAfter z (from prod)\nBefore sum (from prod)\nEnter sum\nBefore x (from sum)\nEnter x\nExit x\nAccept into sum\nAfter x (from sum)\nBefore y (from sum)\nEnter y\nExit y\nAccept into sum\nAfter y (from sum)\nExit sum\nAccept into prod\nAfter sum (from prod)\nExit prod\nAccept into sum\nAfter prod (from sum)\nExit sum\nFinalize')
import logging
import math
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import (
from pyomo.environ import (
import pyomo.repn.util
from pyomo.repn.util import (
def test_BeforeChildDispatcher_registration(self):

    class BeforeChildDispatcherTester(BeforeChildDispatcher):

        @staticmethod
        def _before_var(visitor, child):
            return child

        @staticmethod
        def _before_named_expression(visitor, child):
            return child

    class VisitorTester(object):

        def check_constant(self, value, node):
            return value

        def evaluate(self, node):
            return node()
    visitor = VisitorTester()
    bcd = BeforeChildDispatcherTester()
    self.assertEqual(len(bcd), 0)
    node = 5
    self.assertEqual(bcd[node.__class__](None, node), (False, (_CONSTANT, 5)))
    self.assertIs(bcd[int], bcd._before_native_numeric)
    self.assertEqual(len(bcd), 1)
    node = 'string'
    ans = bcd[node.__class__](None, node)
    self.assertEqual(ans, (False, (_CONSTANT, InvalidNumber(node))))
    self.assertEqual(''.join(ans[1][1].causes), "'string' (str) is not a valid numeric type")
    self.assertIs(bcd[str], bcd._before_string)
    self.assertEqual(len(bcd), 2)
    node = True
    ans = bcd[node.__class__](None, node)
    self.assertEqual(ans, (False, (_CONSTANT, InvalidNumber(node))))
    self.assertEqual(''.join(ans[1][1].causes), 'True (bool) is not a valid numeric type')
    self.assertIs(bcd[bool], bcd._before_native_logical)
    self.assertEqual(len(bcd), 3)
    node = 1j
    ans = bcd[node.__class__](None, node)
    self.assertEqual(ans, (False, (_CONSTANT, InvalidNumber(node))))
    self.assertEqual(''.join(ans[1][1].causes), 'Complex number returned from expression')
    self.assertIs(bcd[complex], bcd._before_complex)
    self.assertEqual(len(bcd), 4)

    class new_int(int):
        pass
    node = new_int(5)
    self.assertEqual(bcd[node.__class__](None, node), (False, (_CONSTANT, 5)))
    self.assertIs(bcd[new_int], bcd._before_native_numeric)
    self.assertEqual(len(bcd), 5)
    node = []
    ans = bcd[node.__class__](None, node)
    self.assertEqual(ans, (False, (_CONSTANT, InvalidNumber([]))))
    self.assertEqual(''.join(ans[1][1].causes), '[] (list) is not a valid numeric type')
    self.assertIs(bcd[list], bcd._before_invalid)
    self.assertEqual(len(bcd), 6)
    node = Var(initialize=7)
    node.construct()
    self.assertIs(bcd[node.__class__](None, node), node)
    self.assertIs(bcd[node.__class__], bcd._before_var)
    self.assertEqual(len(bcd), 7)
    node = Param(initialize=8)
    node.construct()
    self.assertEqual(bcd[node.__class__](visitor, node), (False, (_CONSTANT, 8)))
    self.assertIs(bcd[node.__class__], bcd._before_param)
    self.assertEqual(len(bcd), 8)
    node = Expression(initialize=9)
    node.construct()
    self.assertIs(bcd[node.__class__](None, node), node)
    self.assertIs(bcd[node.__class__], bcd._before_named_expression)
    self.assertEqual(len(bcd), 9)
    node = SumExpression((3, 5))
    self.assertEqual(bcd[node.__class__](None, node), (True, None))
    self.assertIs(bcd[node.__class__], bcd._before_general_expression)
    self.assertEqual(len(bcd), 10)
    node = NPV_ProductExpression((3, 5))
    self.assertEqual(bcd[node.__class__](visitor, node), (False, (_CONSTANT, 15)))
    self.assertEqual(len(bcd), 12)
    self.assertIs(bcd[NPV_ProductExpression], bcd._before_npv)
    self.assertIs(bcd[ProductExpression], bcd._before_general_expression)
    self.assertEqual(len(bcd), 12)
    node = NPV_DivisionExpression((3, 0))
    self.assertEqual(bcd[node.__class__](visitor, node), (True, None))
    self.assertEqual(len(bcd), 14)
    self.assertIs(bcd[NPV_DivisionExpression], bcd._before_npv)
    self.assertIs(bcd[DivisionExpression], bcd._before_general_expression)
    self.assertEqual(len(bcd), 14)
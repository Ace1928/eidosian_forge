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
def test_apply_operation(self):
    m = ConcreteModel()
    m.x = Var()
    div = 1 / m.x
    mul = m.x * m.x
    exp = m.x ** (1 / 2)
    with LoggingIntercept() as LOG:
        self.assertEqual(apply_node_operation(exp, [4, 1 / 2]), 2)
    self.assertEqual(LOG.getvalue(), '')
    with LoggingIntercept() as LOG:
        ans = apply_node_operation(mul, [float('inf'), 0])
        self.assertIs(type(ans), InvalidNumber)
        self.assertTrue(math.isnan(ans.value))
    self.assertEqual(LOG.getvalue(), '')
    _halt = pyomo.repn.util.HALT_ON_EVALUATION_ERROR
    try:
        pyomo.repn.util.HALT_ON_EVALUATION_ERROR = True
        with LoggingIntercept() as LOG:
            with self.assertRaisesRegex(ZeroDivisionError, 'division by zero'):
                apply_node_operation(div, [1, 0])
        self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/x\n")
        pyomo.repn.util.HALT_ON_EVALUATION_ERROR = False
        with LoggingIntercept() as LOG:
            val = apply_node_operation(div, [1, 0])
            self.assertEqual(str(val), 'InvalidNumber(nan)')
        self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/x\n")
    finally:
        pyomo.repn.util.HALT_ON_EVALUATION_ERROR = _halt
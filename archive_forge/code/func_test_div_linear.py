import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_linear(self):
    tests = [(self.linear, self.invalid, NotImplemented), (self.linear, self.asbinary, DivisionExpression((self.linear, self.bin))), (self.linear, self.zero, ZeroDivisionError), (self.linear, self.one, self.linear), (self.linear, self.native, DivisionExpression((self.linear, 5))), (self.linear, self.npv, DivisionExpression((self.linear, self.npv))), (self.linear, self.param, DivisionExpression((self.linear, 6))), (self.linear, self.param_mut, DivisionExpression((self.linear, self.param_mut))), (self.linear, self.var, DivisionExpression((self.linear, self.var))), (self.linear, self.mon_native, DivisionExpression((self.linear, self.mon_native))), (self.linear, self.mon_param, DivisionExpression((self.linear, self.mon_param))), (self.linear, self.mon_npv, DivisionExpression((self.linear, self.mon_npv))), (self.linear, self.linear, DivisionExpression((self.linear, self.linear))), (self.linear, self.sum, DivisionExpression((self.linear, self.sum))), (self.linear, self.other, DivisionExpression((self.linear, self.other))), (self.linear, self.mutable_l0, ZeroDivisionError), (self.linear, self.mutable_l1, DivisionExpression((self.linear, self.mon_npv))), (self.linear, self.mutable_l2, DivisionExpression((self.linear, self.mutable_l2))), (self.linear, self.param0, ZeroDivisionError), (self.linear, self.param1, self.linear), (self.linear, self.mutable_l3, DivisionExpression((self.linear, self.npv)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)
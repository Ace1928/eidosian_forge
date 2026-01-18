import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_param1(self):
    tests = [(self.param1, self.invalid, NotImplemented), (self.param1, self.asbinary, DivisionExpression((1, self.bin))), (self.param1, self.zero, ZeroDivisionError), (self.param1, self.one, 1.0), (self.param1, self.native, 0.2), (self.param1, self.npv, NPV_DivisionExpression((1, self.npv))), (self.param1, self.param, 1 / 6), (self.param1, self.param_mut, NPV_DivisionExpression((1, self.param_mut))), (self.param1, self.var, DivisionExpression((1, self.var))), (self.param1, self.mon_native, DivisionExpression((1, self.mon_native))), (self.param1, self.mon_param, DivisionExpression((1, self.mon_param))), (self.param1, self.mon_npv, DivisionExpression((1, self.mon_npv))), (self.param1, self.linear, DivisionExpression((1, self.linear))), (self.param1, self.sum, DivisionExpression((1, self.sum))), (self.param1, self.other, DivisionExpression((1, self.other))), (self.param1, self.mutable_l0, ZeroDivisionError), (self.param1, self.mutable_l1, DivisionExpression((1, self.mon_npv))), (self.param1, self.mutable_l2, DivisionExpression((1, self.mutable_l2))), (self.param1, self.param0, ZeroDivisionError), (self.param1, self.param1, 1.0), (self.param1, self.mutable_l3, NPV_DivisionExpression((1, self.npv)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)
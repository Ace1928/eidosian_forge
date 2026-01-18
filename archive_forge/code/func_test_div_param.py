import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_param(self):
    tests = [(self.param, self.invalid, NotImplemented), (self.param, self.asbinary, DivisionExpression((6, self.bin))), (self.param, self.zero, ZeroDivisionError), (self.param, self.one, 6.0), (self.param, self.native, 1.2), (self.param, self.npv, NPV_DivisionExpression((6, self.npv))), (self.param, self.param, 1.0), (self.param, self.param_mut, NPV_DivisionExpression((6, self.param_mut))), (self.param, self.var, DivisionExpression((6, self.var))), (self.param, self.mon_native, DivisionExpression((6, self.mon_native))), (self.param, self.mon_param, DivisionExpression((6, self.mon_param))), (self.param, self.mon_npv, DivisionExpression((6, self.mon_npv))), (self.param, self.linear, DivisionExpression((6, self.linear))), (self.param, self.sum, DivisionExpression((6, self.sum))), (self.param, self.other, DivisionExpression((6, self.other))), (self.param, self.mutable_l0, ZeroDivisionError), (self.param, self.mutable_l1, DivisionExpression((6, self.mon_npv))), (self.param, self.mutable_l2, DivisionExpression((6, self.mutable_l2))), (self.param, self.param0, ZeroDivisionError), (self.param, self.param1, 6.0), (self.param, self.mutable_l3, NPV_DivisionExpression((6, self.npv)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)
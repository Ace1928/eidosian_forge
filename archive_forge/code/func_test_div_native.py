import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_native(self):
    tests = [(self.native, self.invalid, NotImplemented), (self.native, self.asbinary, DivisionExpression((5, self.bin))), (self.native, self.zero, ZeroDivisionError), (self.native, self.one, 5.0), (self.native, self.native, 1.0), (self.native, self.npv, NPV_DivisionExpression((5, self.npv))), (self.native, self.param, 5 / 6), (self.native, self.param_mut, NPV_DivisionExpression((5, self.param_mut))), (self.native, self.var, DivisionExpression((5, self.var))), (self.native, self.mon_native, DivisionExpression((5, self.mon_native))), (self.native, self.mon_param, DivisionExpression((5, self.mon_param))), (self.native, self.mon_npv, DivisionExpression((5, self.mon_npv))), (self.native, self.linear, DivisionExpression((5, self.linear))), (self.native, self.sum, DivisionExpression((5, self.sum))), (self.native, self.other, DivisionExpression((5, self.other))), (self.native, self.mutable_l0, ZeroDivisionError), (self.native, self.mutable_l1, DivisionExpression((5, self.mon_npv))), (self.native, self.mutable_l2, DivisionExpression((5, self.mutable_l2))), (self.native, self.param0, ZeroDivisionError), (self.native, self.param1, 5.0), (self.native, self.mutable_l3, NPV_DivisionExpression((5, self.npv)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)
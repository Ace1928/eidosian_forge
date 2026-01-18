import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_mon_native(self):
    tests = [(self.mon_native, self.invalid, NotImplemented), (self.mon_native, self.asbinary, DivisionExpression((self.mon_native, self.bin))), (self.mon_native, self.zero, ZeroDivisionError), (self.mon_native, self.one, self.mon_native), (self.mon_native, self.native, MonomialTermExpression((0.6, self.mon_native.arg(1)))), (self.mon_native, self.npv, MonomialTermExpression((NPV_DivisionExpression((self.mon_native.arg(0), self.npv)), self.mon_native.arg(1)))), (self.mon_native, self.param, MonomialTermExpression((0.5, self.mon_native.arg(1)))), (self.mon_native, self.param_mut, MonomialTermExpression((NPV_DivisionExpression((self.mon_native.arg(0), self.param_mut)), self.mon_native.arg(1)))), (self.mon_native, self.var, DivisionExpression((self.mon_native, self.var))), (self.mon_native, self.mon_native, DivisionExpression((self.mon_native, self.mon_native))), (self.mon_native, self.mon_param, DivisionExpression((self.mon_native, self.mon_param))), (self.mon_native, self.mon_npv, DivisionExpression((self.mon_native, self.mon_npv))), (self.mon_native, self.linear, DivisionExpression((self.mon_native, self.linear))), (self.mon_native, self.sum, DivisionExpression((self.mon_native, self.sum))), (self.mon_native, self.other, DivisionExpression((self.mon_native, self.other))), (self.mon_native, self.mutable_l0, ZeroDivisionError), (self.mon_native, self.mutable_l1, DivisionExpression((self.mon_native, self.mon_npv))), (self.mon_native, self.mutable_l2, DivisionExpression((self.mon_native, self.mutable_l2))), (self.mon_native, self.param0, ZeroDivisionError), (self.mon_native, self.param1, self.mon_native), (self.mon_native, self.mutable_l3, MonomialTermExpression((NPV_DivisionExpression((self.mon_native.arg(0), self.npv)), self.mon_native.arg(1))))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)
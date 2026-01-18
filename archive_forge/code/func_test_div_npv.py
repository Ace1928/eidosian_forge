import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_npv(self):
    tests = [(self.npv, self.invalid, NotImplemented), (self.npv, self.asbinary, DivisionExpression((self.npv, self.bin))), (self.npv, self.zero, ZeroDivisionError), (self.npv, self.one, self.npv), (self.npv, self.native, NPV_DivisionExpression((self.npv, 5))), (self.npv, self.npv, NPV_DivisionExpression((self.npv, self.npv))), (self.npv, self.param, NPV_DivisionExpression((self.npv, 6))), (self.npv, self.param_mut, NPV_DivisionExpression((self.npv, self.param_mut))), (self.npv, self.var, DivisionExpression((self.npv, self.var))), (self.npv, self.mon_native, DivisionExpression((self.npv, self.mon_native))), (self.npv, self.mon_param, DivisionExpression((self.npv, self.mon_param))), (self.npv, self.mon_npv, DivisionExpression((self.npv, self.mon_npv))), (self.npv, self.linear, DivisionExpression((self.npv, self.linear))), (self.npv, self.sum, DivisionExpression((self.npv, self.sum))), (self.npv, self.other, DivisionExpression((self.npv, self.other))), (self.npv, self.mutable_l0, ZeroDivisionError), (self.npv, self.mutable_l1, DivisionExpression((self.npv, self.mon_npv))), (self.npv, self.mutable_l2, DivisionExpression((self.npv, self.mutable_l2))), (self.npv, self.param0, ZeroDivisionError), (self.npv, self.param1, self.npv), (self.npv, self.mutable_l3, NPV_DivisionExpression((self.npv, self.npv)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)
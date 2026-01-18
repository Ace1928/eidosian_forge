import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_param_mut(self):
    tests = [(self.param_mut, self.invalid, NotImplemented), (self.param_mut, self.asbinary, DivisionExpression((self.param_mut, self.bin))), (self.param_mut, self.zero, ZeroDivisionError), (self.param_mut, self.one, self.param_mut), (self.param_mut, self.native, NPV_DivisionExpression((self.param_mut, 5))), (self.param_mut, self.npv, NPV_DivisionExpression((self.param_mut, self.npv))), (self.param_mut, self.param, NPV_DivisionExpression((self.param_mut, 6))), (self.param_mut, self.param_mut, NPV_DivisionExpression((self.param_mut, self.param_mut))), (self.param_mut, self.var, DivisionExpression((self.param_mut, self.var))), (self.param_mut, self.mon_native, DivisionExpression((self.param_mut, self.mon_native))), (self.param_mut, self.mon_param, DivisionExpression((self.param_mut, self.mon_param))), (self.param_mut, self.mon_npv, DivisionExpression((self.param_mut, self.mon_npv))), (self.param_mut, self.linear, DivisionExpression((self.param_mut, self.linear))), (self.param_mut, self.sum, DivisionExpression((self.param_mut, self.sum))), (self.param_mut, self.other, DivisionExpression((self.param_mut, self.other))), (self.param_mut, self.mutable_l0, ZeroDivisionError), (self.param_mut, self.mutable_l1, DivisionExpression((self.param_mut, self.mon_npv))), (self.param_mut, self.mutable_l2, DivisionExpression((self.param_mut, self.mutable_l2))), (self.param_mut, self.param0, ZeroDivisionError), (self.param_mut, self.param1, self.param_mut), (self.param_mut, self.mutable_l3, NPV_DivisionExpression((self.param_mut, self.npv)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)
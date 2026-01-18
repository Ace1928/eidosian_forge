import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_asbinary(self):
    tests = [(self.asbinary, self.invalid, NotImplemented), (self.asbinary, self.asbinary, NotImplemented), (self.asbinary, self.zero, ZeroDivisionError), (self.asbinary, self.one, self.bin), (self.asbinary, self.native, MonomialTermExpression((0.2, self.bin))), (self.asbinary, self.npv, MonomialTermExpression((NPV_DivisionExpression((1, self.npv)), self.bin))), (self.asbinary, self.param, MonomialTermExpression((1 / 6, self.bin))), (self.asbinary, self.param_mut, MonomialTermExpression((NPV_DivisionExpression((1, self.param_mut)), self.bin))), (self.asbinary, self.var, DivisionExpression((self.bin, self.var))), (self.asbinary, self.mon_native, DivisionExpression((self.bin, self.mon_native))), (self.asbinary, self.mon_param, DivisionExpression((self.bin, self.mon_param))), (self.asbinary, self.mon_npv, DivisionExpression((self.bin, self.mon_npv))), (self.asbinary, self.linear, DivisionExpression((self.bin, self.linear))), (self.asbinary, self.sum, DivisionExpression((self.bin, self.sum))), (self.asbinary, self.other, DivisionExpression((self.bin, self.other))), (self.asbinary, self.mutable_l0, ZeroDivisionError), (self.asbinary, self.mutable_l1, DivisionExpression((self.bin, self.mon_npv))), (self.asbinary, self.mutable_l2, DivisionExpression((self.bin, self.mutable_l2))), (self.asbinary, self.param0, ZeroDivisionError), (self.asbinary, self.param1, self.bin), (self.asbinary, self.mutable_l3, MonomialTermExpression((NPV_DivisionExpression((1, self.npv)), self.bin)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)
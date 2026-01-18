import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_mon_npv(self):
    tests = [(self.mon_npv, self.invalid, NotImplemented), (self.mon_npv, self.asbinary, DivisionExpression((self.mon_npv, self.bin))), (self.mon_npv, self.zero, ZeroDivisionError), (self.mon_npv, self.one, self.mon_npv), (self.mon_npv, self.native, MonomialTermExpression((NPV_DivisionExpression((self.mon_npv.arg(0), 5)), self.mon_npv.arg(1)))), (self.mon_npv, self.npv, MonomialTermExpression((NPV_DivisionExpression((self.mon_npv.arg(0), self.npv)), self.mon_npv.arg(1)))), (self.mon_npv, self.param, MonomialTermExpression((NPV_DivisionExpression((self.mon_npv.arg(0), 6)), self.mon_npv.arg(1)))), (self.mon_npv, self.param_mut, MonomialTermExpression((NPV_DivisionExpression((self.mon_npv.arg(0), self.param_mut)), self.mon_npv.arg(1)))), (self.mon_npv, self.var, DivisionExpression((self.mon_npv, self.var))), (self.mon_npv, self.mon_native, DivisionExpression((self.mon_npv, self.mon_native))), (self.mon_npv, self.mon_param, DivisionExpression((self.mon_npv, self.mon_param))), (self.mon_npv, self.mon_npv, DivisionExpression((self.mon_npv, self.mon_npv))), (self.mon_npv, self.linear, DivisionExpression((self.mon_npv, self.linear))), (self.mon_npv, self.sum, DivisionExpression((self.mon_npv, self.sum))), (self.mon_npv, self.other, DivisionExpression((self.mon_npv, self.other))), (self.mon_npv, self.mutable_l0, ZeroDivisionError), (self.mon_npv, self.mutable_l1, DivisionExpression((self.mon_npv, self.mon_npv))), (self.mon_npv, self.mutable_l2, DivisionExpression((self.mon_npv, self.mutable_l2))), (self.mon_npv, self.param0, ZeroDivisionError), (self.mon_npv, self.param1, self.mon_npv), (self.mon_npv, self.mutable_l3, MonomialTermExpression((NPV_DivisionExpression((self.mon_npv.arg(0), self.npv)), self.mon_npv.arg(1))))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)
import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_var(self):
    tests = [(self.var, self.invalid, NotImplemented), (self.var, self.asbinary, DivisionExpression((self.var, self.bin))), (self.var, self.zero, ZeroDivisionError), (self.var, self.one, self.var), (self.var, self.native, MonomialTermExpression((0.2, self.var))), (self.var, self.npv, MonomialTermExpression((NPV_DivisionExpression((1, self.npv)), self.var))), (self.var, self.param, MonomialTermExpression((1 / 6.0, self.var))), (self.var, self.param_mut, MonomialTermExpression((NPV_DivisionExpression((1, self.param_mut)), self.var))), (self.var, self.var, DivisionExpression((self.var, self.var))), (self.var, self.mon_native, DivisionExpression((self.var, self.mon_native))), (self.var, self.mon_param, DivisionExpression((self.var, self.mon_param))), (self.var, self.mon_npv, DivisionExpression((self.var, self.mon_npv))), (self.var, self.linear, DivisionExpression((self.var, self.linear))), (self.var, self.sum, DivisionExpression((self.var, self.sum))), (self.var, self.other, DivisionExpression((self.var, self.other))), (self.var, self.mutable_l0, ZeroDivisionError), (self.var, self.mutable_l1, DivisionExpression((self.var, self.mon_npv))), (self.var, self.mutable_l2, DivisionExpression((self.var, self.mutable_l2))), (self.var, self.param0, ZeroDivisionError), (self.var, self.param1, self.var), (self.var, self.mutable_l3, MonomialTermExpression((NPV_DivisionExpression((1, self.npv)), self.var)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)
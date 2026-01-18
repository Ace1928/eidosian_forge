import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_mon_param(self):
    tests = [(self.mon_param, self.invalid, NotImplemented), (self.mon_param, self.asbinary, ProductExpression((self.mon_param, self.bin))), (self.mon_param, self.zero, 0), (self.mon_param, self.one, self.mon_param), (self.mon_param, self.native, MonomialTermExpression((NPV_ProductExpression((self.mon_param.arg(0), 5)), self.mon_param.arg(1)))), (self.mon_param, self.npv, MonomialTermExpression((NPV_ProductExpression((self.mon_param.arg(0), self.npv)), self.mon_param.arg(1)))), (self.mon_param, self.param, MonomialTermExpression((NPV_ProductExpression((self.mon_param.arg(0), 6)), self.mon_param.arg(1)))), (self.mon_param, self.param_mut, MonomialTermExpression((NPV_ProductExpression((self.mon_param.arg(0), self.param_mut)), self.mon_param.arg(1)))), (self.mon_param, self.var, ProductExpression((self.mon_param, self.var))), (self.mon_param, self.mon_native, ProductExpression((self.mon_param, self.mon_native))), (self.mon_param, self.mon_param, ProductExpression((self.mon_param, self.mon_param))), (self.mon_param, self.mon_npv, ProductExpression((self.mon_param, self.mon_npv))), (self.mon_param, self.linear, ProductExpression((self.mon_param, self.linear))), (self.mon_param, self.sum, ProductExpression((self.mon_param, self.sum))), (self.mon_param, self.other, ProductExpression((self.mon_param, self.other))), (self.mon_param, self.mutable_l0, 0), (self.mon_param, self.mutable_l1, ProductExpression((self.mon_param, self.mon_npv))), (self.mon_param, self.mutable_l2, ProductExpression((self.mon_param, self.mutable_l2))), (self.mon_param, self.param0, 0), (self.mon_param, self.param1, self.mon_param), (self.mon_param, self.mutable_l3, MonomialTermExpression((NPV_ProductExpression((self.mon_param.arg(0), self.npv)), self.mon_param.arg(1))))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)
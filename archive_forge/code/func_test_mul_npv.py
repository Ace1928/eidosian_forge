import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_npv(self):
    tests = [(self.npv, self.invalid, NotImplemented), (self.npv, self.asbinary, MonomialTermExpression((self.npv, self.bin))), (self.npv, self.zero, 0), (self.npv, self.one, self.npv), (self.npv, self.native, NPV_ProductExpression((self.npv, 5))), (self.npv, self.npv, NPV_ProductExpression((self.npv, self.npv))), (self.npv, self.param, NPV_ProductExpression((self.npv, 6))), (self.npv, self.param_mut, NPV_ProductExpression((self.npv, self.param_mut))), (self.npv, self.var, MonomialTermExpression((self.npv, self.var))), (self.npv, self.mon_native, MonomialTermExpression((NPV_ProductExpression((self.npv, self.mon_native.arg(0))), self.mon_native.arg(1)))), (self.npv, self.mon_param, MonomialTermExpression((NPV_ProductExpression((self.npv, self.mon_param.arg(0))), self.mon_param.arg(1)))), (self.npv, self.mon_npv, MonomialTermExpression((NPV_ProductExpression((self.npv, self.mon_npv.arg(0))), self.mon_npv.arg(1)))), (self.npv, self.linear, ProductExpression((self.npv, self.linear))), (self.npv, self.sum, ProductExpression((self.npv, self.sum))), (self.npv, self.other, ProductExpression((self.npv, self.other))), (self.npv, self.mutable_l0, 0), (self.npv, self.mutable_l1, MonomialTermExpression((NPV_ProductExpression((self.npv, self.mon_npv.arg(0))), self.mon_npv.arg(1)))), (self.npv, self.mutable_l2, ProductExpression((self.npv, self.mutable_l2))), (self.npv, self.param0, 0), (self.npv, self.param1, self.npv), (self.npv, self.mutable_l3, NPV_ProductExpression((self.npv, self.npv)))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)
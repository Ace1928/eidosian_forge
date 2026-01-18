import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_param_mut(self):
    tests = [(self.param_mut, self.invalid, NotImplemented), (self.param_mut, self.asbinary, MonomialTermExpression((self.param_mut, self.bin))), (self.param_mut, self.zero, 0), (self.param_mut, self.one, self.param_mut), (self.param_mut, self.native, NPV_ProductExpression((self.param_mut, 5))), (self.param_mut, self.npv, NPV_ProductExpression((self.param_mut, self.npv))), (self.param_mut, self.param, NPV_ProductExpression((self.param_mut, 6))), (self.param_mut, self.param_mut, NPV_ProductExpression((self.param_mut, self.param_mut))), (self.param_mut, self.var, MonomialTermExpression((self.param_mut, self.var))), (self.param_mut, self.mon_native, MonomialTermExpression((NPV_ProductExpression((self.param_mut, self.mon_native.arg(0))), self.mon_native.arg(1)))), (self.param_mut, self.mon_param, MonomialTermExpression((NPV_ProductExpression((self.param_mut, self.mon_param.arg(0))), self.mon_param.arg(1)))), (self.param_mut, self.mon_npv, MonomialTermExpression((NPV_ProductExpression((self.param_mut, self.mon_npv.arg(0))), self.mon_npv.arg(1)))), (self.param_mut, self.linear, ProductExpression((self.param_mut, self.linear))), (self.param_mut, self.sum, ProductExpression((self.param_mut, self.sum))), (self.param_mut, self.other, ProductExpression((self.param_mut, self.other))), (self.param_mut, self.mutable_l0, 0), (self.param_mut, self.mutable_l1, MonomialTermExpression((NPV_ProductExpression((self.param_mut, self.mon_npv.arg(0))), self.mon_npv.arg(1)))), (self.param_mut, self.mutable_l2, ProductExpression((self.param_mut, self.mutable_l2))), (self.param_mut, self.param0, 0), (self.param_mut, self.param1, self.param_mut), (self.param_mut, self.mutable_l3, NPV_ProductExpression((self.param_mut, self.npv)))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)
import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_other(self):
    tests = [(self.other, self.invalid, NotImplemented), (self.other, self.asbinary, ProductExpression((self.other, self.bin))), (self.other, self.zero, 0), (self.other, self.one, self.other), (self.other, self.native, ProductExpression((self.other, 5))), (self.other, self.npv, ProductExpression((self.other, self.npv))), (self.other, self.param, ProductExpression((self.other, 6))), (self.other, self.param_mut, ProductExpression((self.other, self.param_mut))), (self.other, self.var, ProductExpression((self.other, self.var))), (self.other, self.mon_native, ProductExpression((self.other, self.mon_native))), (self.other, self.mon_param, ProductExpression((self.other, self.mon_param))), (self.other, self.mon_npv, ProductExpression((self.other, self.mon_npv))), (self.other, self.linear, ProductExpression((self.other, self.linear))), (self.other, self.sum, ProductExpression((self.other, self.sum))), (self.other, self.other, ProductExpression((self.other, self.other))), (self.other, self.mutable_l0, 0), (self.other, self.mutable_l1, ProductExpression((self.other, self.mon_npv))), (self.other, self.mutable_l2, ProductExpression((self.other, self.mutable_l2))), (self.other, self.param0, 0), (self.other, self.param1, self.other), (self.other, self.mutable_l3, ProductExpression((self.other, self.npv)))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)
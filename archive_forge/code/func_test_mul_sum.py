import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_sum(self):
    tests = [(self.sum, self.invalid, NotImplemented), (self.sum, self.asbinary, ProductExpression((self.sum, self.bin))), (self.sum, self.zero, 0), (self.sum, self.one, self.sum), (self.sum, self.native, ProductExpression((self.sum, 5))), (self.sum, self.npv, ProductExpression((self.sum, self.npv))), (self.sum, self.param, ProductExpression((self.sum, 6))), (self.sum, self.param_mut, ProductExpression((self.sum, self.param_mut))), (self.sum, self.var, ProductExpression((self.sum, self.var))), (self.sum, self.mon_native, ProductExpression((self.sum, self.mon_native))), (self.sum, self.mon_param, ProductExpression((self.sum, self.mon_param))), (self.sum, self.mon_npv, ProductExpression((self.sum, self.mon_npv))), (self.sum, self.linear, ProductExpression((self.sum, self.linear))), (self.sum, self.sum, ProductExpression((self.sum, self.sum))), (self.sum, self.other, ProductExpression((self.sum, self.other))), (self.sum, self.mutable_l0, 0), (self.sum, self.mutable_l1, ProductExpression((self.sum, self.mon_npv))), (self.sum, self.mutable_l2, ProductExpression((self.sum, self.mutable_l2))), (self.sum, self.param0, 0), (self.sum, self.param1, self.sum), (self.sum, self.mutable_l3, ProductExpression((self.sum, self.npv)))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)
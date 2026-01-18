import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_linear(self):
    tests = [(self.linear, self.invalid, NotImplemented), (self.linear, self.asbinary, ProductExpression((self.linear, self.bin))), (self.linear, self.zero, 0), (self.linear, self.one, self.linear), (self.linear, self.native, ProductExpression((self.linear, 5))), (self.linear, self.npv, ProductExpression((self.linear, self.npv))), (self.linear, self.param, ProductExpression((self.linear, 6))), (self.linear, self.param_mut, ProductExpression((self.linear, self.param_mut))), (self.linear, self.var, ProductExpression((self.linear, self.var))), (self.linear, self.mon_native, ProductExpression((self.linear, self.mon_native))), (self.linear, self.mon_param, ProductExpression((self.linear, self.mon_param))), (self.linear, self.mon_npv, ProductExpression((self.linear, self.mon_npv))), (self.linear, self.linear, ProductExpression((self.linear, self.linear))), (self.linear, self.sum, ProductExpression((self.linear, self.sum))), (self.linear, self.other, ProductExpression((self.linear, self.other))), (self.linear, self.mutable_l0, 0), (self.linear, self.mutable_l1, ProductExpression((self.linear, self.mon_npv))), (self.linear, self.mutable_l2, ProductExpression((self.linear, self.mutable_l2))), (self.linear, self.param0, 0), (self.linear, self.param1, self.linear), (self.linear, self.mutable_l3, ProductExpression((self.linear, self.npv)))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)
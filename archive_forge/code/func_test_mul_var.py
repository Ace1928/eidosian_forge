import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_var(self):
    tests = [(self.var, self.invalid, NotImplemented), (self.var, self.asbinary, ProductExpression((self.var, self.bin))), (self.var, self.zero, 0), (self.var, self.one, self.var), (self.var, self.native, MonomialTermExpression((5, self.var))), (self.var, self.npv, MonomialTermExpression((self.npv, self.var))), (self.var, self.param, MonomialTermExpression((6, self.var))), (self.var, self.param_mut, MonomialTermExpression((self.param_mut, self.var))), (self.var, self.var, ProductExpression((self.var, self.var))), (self.var, self.mon_native, ProductExpression((self.var, self.mon_native))), (self.var, self.mon_param, ProductExpression((self.var, self.mon_param))), (self.var, self.mon_npv, ProductExpression((self.var, self.mon_npv))), (self.var, self.linear, ProductExpression((self.var, self.linear))), (self.var, self.sum, ProductExpression((self.var, self.sum))), (self.var, self.other, ProductExpression((self.var, self.other))), (self.var, self.mutable_l0, 0), (self.var, self.mutable_l1, ProductExpression((self.var, self.mon_npv))), (self.var, self.mutable_l2, ProductExpression((self.var, self.mutable_l2))), (self.var, self.param0, 0), (self.var, self.param1, self.var), (self.var, self.mutable_l3, MonomialTermExpression((self.npv, self.var)))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)
import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_neg_neg(self):

    def _neg_neg(x):
        return operator.neg(operator.neg(x))
    tests = [(self.invalid, NotImplemented), (self.asbinary, MonomialTermExpression((1, self.bin))), (self.zero, 0), (self.one, 1), (self.native, 5), (self.npv, self.npv), (self.param, 6), (self.param_mut, self.param_mut), (self.var, MonomialTermExpression((1, self.var))), (self.mon_native, self.mon_native), (self.mon_param, self.mon_param), (self.mon_npv, self.mon_npv), (self.linear, self.linear), (self.sum, self.sum), (self.other, self.other), (self.mutable_l0, 0), (self.mutable_l1, self.mon_npv), (self.mutable_l2, self.mutable_l2), (self.param0, 0), (self.param1, 1), (self.mutable_l3, self.npv)]
    self._run_cases(tests, _neg_neg)
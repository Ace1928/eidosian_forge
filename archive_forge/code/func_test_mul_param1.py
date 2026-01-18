import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_param1(self):
    tests = [(self.param1, self.invalid, self.SKIP), (self.param1, self.asbinary, self.bin), (self.param1, self.zero, 0), (self.param1, self.one, 1), (self.param1, self.native, 5), (self.param1, self.npv, self.npv), (self.param1, self.param, self.param), (self.param1, self.param_mut, self.param_mut), (self.param1, self.var, self.var), (self.param1, self.mon_native, self.mon_native), (self.param1, self.mon_param, self.mon_param), (self.param1, self.mon_npv, self.mon_npv), (self.param1, self.linear, self.linear), (self.param1, self.sum, self.sum), (self.param1, self.other, self.other), (self.param1, self.mutable_l0, 0), (self.param1, self.mutable_l1, self.mon_npv), (self.param1, self.mutable_l2, self.mutable_l2), (self.param1, self.param0, self.param0), (self.param1, self.param1, self.param1), (self.param1, self.mutable_l3, self.npv)]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)
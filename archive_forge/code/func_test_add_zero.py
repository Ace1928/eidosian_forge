import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_zero(self):
    tests = [(self.zero, self.invalid, NotImplemented), (self.zero, self.asbinary, self.bin), (self.zero, self.zero, 0), (self.zero, self.one, 1), (self.zero, self.native, 5), (self.zero, self.npv, self.npv), (self.zero, self.param, 6), (self.zero, self.param_mut, self.param_mut), (self.zero, self.var, self.var), (self.zero, self.mon_native, self.mon_native), (self.zero, self.mon_param, self.mon_param), (self.zero, self.mon_npv, self.mon_npv), (self.zero, self.linear, self.linear), (self.zero, self.sum, self.sum), (self.zero, self.other, self.other), (self.zero, self.mutable_l0, 0), (self.zero, self.mutable_l1, self.mon_npv), (self.zero, self.mutable_l2, self.mutable_l2), (self.zero, self.param0, 0), (self.zero, self.param1, 1), (self.zero, self.mutable_l3, self.npv)]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)
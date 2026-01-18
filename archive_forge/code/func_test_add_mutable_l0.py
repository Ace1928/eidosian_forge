import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_mutable_l0(self):
    tests = [(self.mutable_l0, self.invalid, NotImplemented), (self.mutable_l0, self.asbinary, self.bin), (self.mutable_l0, self.zero, 0), (self.mutable_l0, self.one, 1), (self.mutable_l0, self.native, 5), (self.mutable_l0, self.npv, self.npv), (self.mutable_l0, self.param, 6), (self.mutable_l0, self.param_mut, self.param_mut), (self.mutable_l0, self.var, self.var), (self.mutable_l0, self.mon_native, self.mon_native), (self.mutable_l0, self.mon_param, self.mon_param), (self.mutable_l0, self.mon_npv, self.mon_npv), (self.mutable_l0, self.linear, self.linear), (self.mutable_l0, self.sum, self.sum), (self.mutable_l0, self.other, self.other), (self.mutable_l0, self.mutable_l0, 0), (self.mutable_l0, self.mutable_l1, self.mon_npv), (self.mutable_l0, self.mutable_l2, self.mutable_l2), (self.mutable_l0, self.param0, 0), (self.mutable_l0, self.param1, 1), (self.mutable_l0, self.mutable_l3, self.npv)]
    self._run_cases(tests, operator.add)
import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_native(self):
    tests = [(self.native, self.invalid, NotImplemented), (self.native, self.asbinary, LinearExpression([5, self.mon_bin])), (self.native, self.zero, 5), (self.native, self.one, 6), (self.native, self.native, 10), (self.native, self.npv, NPV_SumExpression([5, self.npv])), (self.native, self.param, 11), (self.native, self.param_mut, NPV_SumExpression([5, self.param_mut])), (self.native, self.var, LinearExpression([5, self.mon_var])), (self.native, self.mon_native, LinearExpression([5, self.mon_native])), (self.native, self.mon_param, LinearExpression([5, self.mon_param])), (self.native, self.mon_npv, LinearExpression([5, self.mon_npv])), (self.native, self.linear, LinearExpression(self.linear.args + [5])), (self.native, self.sum, SumExpression(self.sum.args + [5])), (self.native, self.other, SumExpression([5, self.other])), (self.native, self.mutable_l0, 5), (self.native, self.mutable_l1, LinearExpression([5] + self.mutable_l1.args)), (self.native, self.mutable_l2, SumExpression(self.mutable_l2.args + [5])), (self.native, self.param0, 5), (self.native, self.param1, 6), (self.native, self.mutable_l3, NPV_SumExpression([5, self.npv]))]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)
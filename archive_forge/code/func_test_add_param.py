import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_param(self):
    tests = [(self.param, self.invalid, NotImplemented), (self.param, self.asbinary, LinearExpression([6, self.mon_bin])), (self.param, self.zero, 6), (self.param, self.one, 7), (self.param, self.native, 11), (self.param, self.npv, NPV_SumExpression([6, self.npv])), (self.param, self.param, 12), (self.param, self.param_mut, NPV_SumExpression([6, self.param_mut])), (self.param, self.var, LinearExpression([6, self.mon_var])), (self.param, self.mon_native, LinearExpression([6, self.mon_native])), (self.param, self.mon_param, LinearExpression([6, self.mon_param])), (self.param, self.mon_npv, LinearExpression([6, self.mon_npv])), (self.param, self.linear, LinearExpression(self.linear.args + [6])), (self.param, self.sum, SumExpression(self.sum.args + [6])), (self.param, self.other, SumExpression([6, self.other])), (self.param, self.mutable_l0, 6), (self.param, self.mutable_l1, LinearExpression([6] + self.mutable_l1.args)), (self.param, self.mutable_l2, SumExpression(self.mutable_l2.args + [6])), (self.param, self.param0, 6), (self.param, self.param1, 7), (self.param, self.mutable_l3, NPV_SumExpression([6, self.npv]))]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)
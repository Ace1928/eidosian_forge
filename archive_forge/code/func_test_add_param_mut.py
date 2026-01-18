import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_param_mut(self):
    tests = [(self.param_mut, self.invalid, NotImplemented), (self.param_mut, self.asbinary, LinearExpression([self.param_mut, self.mon_bin])), (self.param_mut, self.zero, self.param_mut), (self.param_mut, self.one, NPV_SumExpression([self.param_mut, 1])), (self.param_mut, self.native, NPV_SumExpression([self.param_mut, 5])), (self.param_mut, self.npv, NPV_SumExpression([self.param_mut, self.npv])), (self.param_mut, self.param, NPV_SumExpression([self.param_mut, 6])), (self.param_mut, self.param_mut, NPV_SumExpression([self.param_mut, self.param_mut])), (self.param_mut, self.var, LinearExpression([self.param_mut, self.mon_var])), (self.param_mut, self.mon_native, LinearExpression([self.param_mut, self.mon_native])), (self.param_mut, self.mon_param, LinearExpression([self.param_mut, self.mon_param])), (self.param_mut, self.mon_npv, LinearExpression([self.param_mut, self.mon_npv])), (self.param_mut, self.linear, LinearExpression(self.linear.args + [self.param_mut])), (self.param_mut, self.sum, SumExpression(self.sum.args + [self.param_mut])), (self.param_mut, self.other, SumExpression([self.param_mut, self.other])), (self.param_mut, self.mutable_l0, self.param_mut), (self.param_mut, self.mutable_l1, LinearExpression([self.param_mut] + self.mutable_l1.args)), (self.param_mut, self.mutable_l2, SumExpression(self.mutable_l2.args + [self.param_mut])), (self.param_mut, self.param0, self.param_mut), (self.param_mut, self.param1, NPV_SumExpression([self.param_mut, 1])), (self.param_mut, self.mutable_l3, NPV_SumExpression([self.param_mut, self.npv]))]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)
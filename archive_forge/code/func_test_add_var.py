import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_var(self):
    tests = [(self.var, self.invalid, NotImplemented), (self.var, self.asbinary, LinearExpression([self.mon_var, self.mon_bin])), (self.var, self.zero, self.var), (self.var, self.one, LinearExpression([self.mon_var, 1])), (self.var, self.native, LinearExpression([self.mon_var, 5])), (self.var, self.npv, LinearExpression([self.mon_var, self.npv])), (self.var, self.param, LinearExpression([self.mon_var, 6])), (self.var, self.param_mut, LinearExpression([self.mon_var, self.param_mut])), (self.var, self.var, LinearExpression([self.mon_var, self.mon_var])), (self.var, self.mon_native, LinearExpression([self.mon_var, self.mon_native])), (self.var, self.mon_param, LinearExpression([self.mon_var, self.mon_param])), (self.var, self.mon_npv, LinearExpression([self.mon_var, self.mon_npv])), (self.var, self.linear, LinearExpression(self.linear.args + [self.mon_var])), (self.var, self.sum, SumExpression(self.sum.args + [self.var])), (self.var, self.other, SumExpression([self.var, self.other])), (self.var, self.mutable_l0, self.var), (self.var, self.mutable_l1, LinearExpression([self.mon_var] + self.mutable_l1.args)), (self.var, self.mutable_l2, SumExpression(self.mutable_l2.args + [self.var])), (self.var, self.param0, self.var), (self.var, self.param1, LinearExpression([self.mon_var, 1])), (self.var, self.mutable_l3, LinearExpression([MonomialTermExpression((1, self.var)), self.npv]))]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)
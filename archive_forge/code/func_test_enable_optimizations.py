import logging
import math
import operator
import sys
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from pyomo.core.expr.visitor import clone_expression
from pyomo.environ import ConcreteModel, Param, Var, BooleanVar
from pyomo.gdp import Disjunct
def test_enable_optimizations(self):
    enable_expression_optimizations(zero=False, one=False)
    self.assertEqual(_zero_one_optimizations, set())
    enable_expression_optimizations(zero=True, one=False)
    self.assertEqual(_zero_one_optimizations, {0})
    enable_expression_optimizations(zero=True, one=True)
    self.assertEqual(_zero_one_optimizations, {0, 1})
    enable_expression_optimizations(zero=False, one=True)
    self.assertEqual(_zero_one_optimizations, {1})
    enable_expression_optimizations()
    self.assertEqual(_zero_one_optimizations, {1})
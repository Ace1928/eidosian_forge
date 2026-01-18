import logging
import math
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.compare import assertExpressionsStructurallyEqual
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.environ import ConcreteModel, Param, Var, ExternalFunction
def test_mutable_expression(self):
    m = ConcreteModel()
    m.x = Var(range(3))
    with mutable_expression() as e:
        f = e
        self.assertIs(type(e), _MutableNPVSumExpression)
        e += 1
        self.assertIs(e, f)
        self.assertIs(type(e), _MutableNPVSumExpression)
        e += m.x[0]
        self.assertIs(e, f)
        self.assertIs(type(e), _MutableLinearExpression)
        e += 100 * m.x[1]
        self.assertIs(e, f)
        self.assertIs(type(e), _MutableLinearExpression)
        e += m.x[0] ** 2
        self.assertIs(e, f)
        self.assertIs(type(e), _MutableSumExpression)
    self.assertIs(e, f)
    self.assertIs(type(e), SumExpression)
import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
def test_identify_mutable_parameters_params(self):
    m = ConcreteModel()
    m.I = RangeSet(3)
    m.a = Param(initialize=1, mutable=True)
    m.b = Param(m.I, initialize=1, mutable=True)
    m.p = Var(initialize=1)
    m.x = ExternalFunction(library='foo.so', function='bar')
    self.assertEqual(list(identify_mutable_parameters(m.a)), [m.a])
    self.assertEqual(list(identify_mutable_parameters(m.b[1])), [m.b[1]])
    self.assertEqual(list(identify_mutable_parameters(m.a + m.b[1])), [m.a, m.b[1]])
    self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1])), [m.a, m.b[1]])
    self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2])), [m.b[2], m.a, m.b[1]])
    self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2] * m.b[3] * m.b[2])), [m.a, m.b[1], m.b[2], m.b[3]])
    self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2] / m.b[3] * m.b[2])), [m.a, m.b[1], m.b[2], m.b[3]])
    self.assertEqual(list(identify_mutable_parameters(m.x(m.a, 'string_param', 1, []) * m.b[1])), [m.b[1], m.a])
    self.assertEqual(list(identify_mutable_parameters(m.x(m.p, 'string_param', 1, []) * m.b[1])), [m.b[1]])
    self.assertEqual(list(identify_mutable_parameters(tanh(m.a) * m.b[1])), [m.b[1], m.a])
    self.assertEqual(list(identify_mutable_parameters(abs(m.a) * m.b[1])), [m.b[1], m.a])
    self.assertEqual(list(identify_mutable_parameters(m.a ** m.a + m.a)), [m.a])
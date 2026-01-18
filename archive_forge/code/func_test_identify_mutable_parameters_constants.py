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
def test_identify_mutable_parameters_constants(self):
    m = ConcreteModel()
    m.x = Var(initialize=1)
    m.x.fix()
    m.p = Param(initialize=2, mutable=False)
    m.p_m = Param(initialize=3, mutable=True)
    e1 = m.x + m.p + NumericConstant(5)
    self.assertEqual(list(identify_mutable_parameters(e1)), [])
    e2 = 5 * m.x + NumericConstant(3) * m.p_m + m.p == 0
    mut_params = list(identify_mutable_parameters(e2))
    self.assertEqual(len(mut_params), 1)
    self.assertIs(mut_params[0], m.p_m)
import pyomo.common.unittest as unittest
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
from io import StringIO
def test_backmap(self):
    m = _generate_boolean_model(3)
    TransformationFactory('core.logical_to_linear').apply_to(m)
    m.Y_asbinary[1].value = 1
    m.Y_asbinary[2].value = 0
    update_boolean_vars_from_binary(m)
    self.assertTrue(m.Y[1].value)
    self.assertFalse(m.Y[2].value)
    self.assertIsNone(m.Y[3].value)
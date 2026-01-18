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
def test_cannot_reassociate_boolean_error(self):
    m = _generate_boolean_model(2)
    TransformationFactory('core.logical_to_linear').apply_to(m)
    m.y = Var(domain=Binary)
    with self.assertRaisesRegex(RuntimeError, "Reassociating BooleanVar 'Y\\[1\\]' \\(currently associated with 'Y_asbinary\\[1\\]'\\) with 'y' is not allowed"):
        m.Y[1].associate_binary_var(m.y)
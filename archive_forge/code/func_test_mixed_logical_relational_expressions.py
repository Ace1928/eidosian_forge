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
def test_mixed_logical_relational_expressions(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = BooleanVar([1, 2])
    m.c = LogicalConstraint(expr=land(m.y[1], m.y[2]).implies(m.x >= 0))
    with self.assertRaisesRegex(MouseTrap, "core.logical_to_linear does not support transforming LogicalConstraints with embedded relational expressions. Found '0.0 <= x'.", normalize_whitespace=True):
        TransformationFactory('core.logical_to_linear').apply_to(m)
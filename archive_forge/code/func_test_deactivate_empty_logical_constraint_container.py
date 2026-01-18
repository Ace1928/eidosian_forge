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
def test_deactivate_empty_logical_constraint_container(self):
    m = ConcreteModel()
    m.propositions = LogicalConstraintList()
    TransformationFactory('core.logical_to_linear').apply_to(m)
    self.assertIsNone(m.component('logic_to_linear'))
    self.assertFalse(m.propositions.active)
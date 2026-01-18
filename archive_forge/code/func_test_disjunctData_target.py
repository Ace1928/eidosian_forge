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
def test_disjunctData_target(self):
    m = ConcreteModel()
    m.d = Disjunct([1, 2])
    m.d[1].transfer_attributes_from(self.make_indexed_logical_constraint_model())
    TransformationFactory('core.logical_to_linear').apply_to(m, targets=m.d[1])
    _constrs_contained_within(self, [(2, m.d[1].Y[1].get_associated_binary() + m.d[1].Y[2].get_associated_binary() + m.d[1].Y[3].get_associated_binary(), 2)], m.d[1].logic_to_linear.transformed_constraints)
    _constrs_contained_within(self, [(1, m.d[1].Y[2].get_associated_binary() + m.d[1].Y[3].get_associated_binary() + (1 - m.d[1].Y[1].get_associated_binary()), None)], m.d[1].logic_to_linear.transformed_constraints)
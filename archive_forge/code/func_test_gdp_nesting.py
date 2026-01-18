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
def test_gdp_nesting(self):
    m = _generate_boolean_model(2)
    m.disj = Disjunction(expr=[[m.Y[1].implies(m.Y[2])], [m.Y[2].equivalent_to(False)]])
    TransformationFactory('core.logical_to_linear').apply_to(m, targets=[m.disj.disjuncts[0], m.disj.disjuncts[1]])
    _constrs_contained_within(self, [(1, 1 - m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary(), None)], m.disj_disjuncts[0].logic_to_linear.transformed_constraints)
    _constrs_contained_within(self, [(1, 1 - m.Y[2].get_associated_binary(), 1)], m.disj_disjuncts[1].logic_to_linear.transformed_constraints)
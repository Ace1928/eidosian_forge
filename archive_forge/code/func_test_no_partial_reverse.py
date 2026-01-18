from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def test_no_partial_reverse(self):
    m = self.make_two_term_disjunction()
    self.add_three_term_disjunction(m)
    m.d1.indicator_var.set_value(True)
    m.d2.indicator_var.set_value(False)
    m.d[2].indicator_var = True
    m.d[3].indicator_var = False
    reverse = TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)
    self.check_fixed_mip(m)
    with self.assertRaisesRegex(ValueError, "The 'gdp.transform_current_disjunctive_state' transformation cannot be called with both targets and a reverse token specified. If reversing the transformation, do not include targets: The reverse transformation will restore all the components the original transformation call transformed."):
        TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m, reverse=reverse, targets=m.disjunction2)
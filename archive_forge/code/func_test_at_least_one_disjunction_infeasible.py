from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def test_at_least_one_disjunction_infeasible(self):
    m = ConcreteModel()
    self.add_three_term_disjunction(m, exactly_one=False)
    m.d[1].indicator_var.fix(False)
    m.d[2].indicator_var = False
    m.d[3].indicator_var = False
    with self.assertRaisesRegex(InfeasibleConstraintException, "Logical constraint for Disjunction 'disjunction2' is violated: All the Disjunct indicator_vars are 'False.'"):
        TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)
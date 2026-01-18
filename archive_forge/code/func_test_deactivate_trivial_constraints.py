import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import Constraint, ConcreteModel, TransformationFactory, Var
def test_deactivate_trivial_constraints(self):
    """Test for deactivation of trivial constraints."""
    m = ConcreteModel()
    m.v1 = Var(initialize=1)
    m.v2 = Var(initialize=2)
    m.v3 = Var(initialize=3)
    m.c = Constraint(expr=m.v1 <= m.v2)
    m.c2 = Constraint(expr=m.v2 >= m.v3)
    m.c3 = Constraint(expr=m.v1 <= 5)
    m.v1.fix()
    TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)
    self.assertTrue(m.c.active)
    self.assertTrue(m.c2.active)
    self.assertFalse(m.c3.active)
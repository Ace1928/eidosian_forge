import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import Constraint, ConcreteModel, TransformationFactory, Var
def test_higher_degree_trivial_constraint(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.c = Constraint(expr=(m.x ** 2 + m.y) * m.z >= -8)
    m.z.fix(0)
    TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)
    self.assertFalse(m.c.active)
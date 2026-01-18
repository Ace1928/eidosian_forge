import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Constraint, TransformationFactory, Var, value
def test_ignore_nonlinear(self):
    m = ConcreteModel()
    m.v1 = Var()
    m.c1 = Constraint(expr=m.v1 * m.v1 >= 2)
    self.assertEqual(value(m.c1.lower), 2)
    self.assertFalse(m.c1.has_ub())
    TransformationFactory('core.tighten_constraints_from_vars').apply_to(m)
    self.assertEqual(value(m.c1.lower), 2)
    self.assertFalse(m.c1.has_ub())
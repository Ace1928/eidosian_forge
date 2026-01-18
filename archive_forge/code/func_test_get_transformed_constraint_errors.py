from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_get_transformed_constraint_errors(self):
    m = self.create_two_disjunction_model()
    m.z = Var()
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m, targets=m.outer)
    out = StringIO()
    with LoggingIntercept(out, 'pyomo.gdp.plugins.bound_pretransformation', logging.DEBUG):
        nothing = bt.get_transformed_constraints(m.z, m.outer)
    self.assertEqual(len(nothing), 0)
    self.assertEqual(out.getvalue(), "Constraint bounding variable 'z' on Disjunction 'outer' was not transformed by the 'gdp.bound_pretransformation' transformation\n")
    out = StringIO()
    with LoggingIntercept(out, 'pyomo.gdp.plugins.bound_pretransformation', logging.DEBUG):
        nothing = bt.get_transformed_constraints(m.x, m.disjunction)
    self.assertEqual(len(nothing), 0)
    self.assertEqual(out.getvalue(), "No variable on Disjunction 'disjunction' was transformed with the gdp.bound_pretransformation transformation\n")
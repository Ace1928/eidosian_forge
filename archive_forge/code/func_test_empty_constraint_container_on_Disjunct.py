from io import StringIO
import logging
from os.path import join, normpath
import pickle
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.tests.common_tests import (
from pyomo.gdp.tests.models import make_indexed_equality_model
from pyomo.repn import generate_standard_repn
def test_empty_constraint_container_on_Disjunct(self):
    m = ConcreteModel()
    m.d = Disjunct()
    m.e = Disjunct()
    m.d.c = Constraint(['s', 'i', 'l', 'L', 'y'])
    m.x = Var(bounds=(2, 3))
    m.e.c = Constraint(expr=m.x == 2.7)
    m.disjunction = Disjunction(expr=[m.d, m.e])
    mbm = TransformationFactory('gdp.mbigm')
    mbm.apply_to(m)
    cons = mbm.get_transformed_constraints(m.e.c)
    self.assertEqual(len(cons), 2)
    self.check_pretty_bound_constraints(cons[0], m.x, {m.d: 2, m.e: 2.7}, lb=True)
    self.check_pretty_bound_constraints(cons[1], m.x, {m.d: 3, m.e: 2.7}, lb=False)
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
def test_local_var_suffix_ignored(self):
    m = self.make_model()
    m.y = Var(bounds=(2, 5))
    m.d1.another_thing = Constraint(expr=m.y == 3)
    m.d1.LocalVars = Suffix(direction=Suffix.LOCAL)
    m.d1.LocalVars[m.d1] = m.y
    mbigm = TransformationFactory('gdp.mbigm')
    mbigm.apply_to(m, reduce_bound_constraints=True, only_mbigm_bound_constraints=True)
    cons = mbigm.get_transformed_constraints(m.d1.x1_bounds)
    self.check_pretty_bound_constraints(cons[0], m.x1, {m.d1: 0.5, m.d2: 0.65, m.d3: 2}, lb=True)
    self.check_pretty_bound_constraints(cons[1], m.x1, {m.d1: 2, m.d2: 3, m.d3: 10}, lb=False)
    cons = mbigm.get_transformed_constraints(m.d1.x2_bounds)
    self.check_pretty_bound_constraints(cons[0], m.x2, {m.d1: 0.75, m.d2: 3, m.d3: 0.55}, lb=True)
    self.check_pretty_bound_constraints(cons[1], m.x2, {m.d1: 3, m.d2: 10, m.d3: 1}, lb=False)
    cons = mbigm.get_transformed_constraints(m.d1.another_thing)
    self.assertEqual(len(cons), 2)
    self.check_pretty_bound_constraints(cons[0], m.y, {m.d1: 3, m.d2: 2, m.d3: 2}, lb=True)
    self.check_pretty_bound_constraints(cons[1], m.y, {m.d1: 3, m.d2: 5, m.d3: 5}, lb=False)
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
def test_Ms_specified_as_args_honored(self):
    m = self.make_model()
    Ms = self.get_Ms(m)
    Ms[m.d2.x2_bounds, m.d3] = (-100, 100)
    Ms[m.d3.func, m.d1] = [10, 20]
    mbigm = TransformationFactory('gdp.mbigm')
    mbigm.apply_to(m, bigM=Ms, reduce_bound_constraints=False)
    self.assertStructuredAlmostEqual(mbigm.get_all_M_values(m), Ms)
    self.check_linear_func_constraints(m, mbigm, Ms)
    cons = mbigm.get_transformed_constraints(m.d2.x2_bounds)
    self.assertEqual(len(cons), 2)
    self.check_untightened_bounds_constraint(cons[0], m.x2, m.d2, m.disjunction, {m.d1: 0.75, m.d3: -97}, lower=3)
    self.check_untightened_bounds_constraint(cons[1], m.x2, m.d2, m.disjunction, {m.d1: 3, m.d3: 110}, upper=10)
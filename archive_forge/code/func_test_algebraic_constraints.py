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
def test_algebraic_constraints(self):
    m = self.make_model()
    mbm = TransformationFactory('gdp.mbigm')
    mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=False)
    self.assertIsNotNone(m.disjunction.algebraic_constraint)
    xor = m.disjunction.algebraic_constraint
    self.assertIs(mbm.get_src_disjunction(xor), m.disjunction)
    self.assertEqual(value(xor.lower), 1)
    self.assertEqual(value(xor.upper), 1)
    repn = generate_standard_repn(xor.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(value(repn.constant), 0)
    self.assertEqual(len(repn.linear_vars), 3)
    check_linear_coef(self, repn, m.d1.binary_indicator_var, 1)
    check_linear_coef(self, repn, m.d2.binary_indicator_var, 1)
    check_linear_coef(self, repn, m.d3.binary_indicator_var, 1)
    check_obj_in_active_tree(self, xor)
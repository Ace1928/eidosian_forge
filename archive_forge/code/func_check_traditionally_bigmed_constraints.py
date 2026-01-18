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
def check_traditionally_bigmed_constraints(self, m, mbm, Ms):
    cons = mbm.get_transformed_constraints(m.d1.func)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    ub = cons[1]
    assertExpressionsEqual(self, lb.expr, 0.0 <= m.x1 + m.x2 - m.d - Ms[m.d1][0] * (1 - m.d1.binary_indicator_var))
    self.assertIsNone(ub.lower)
    self.assertEqual(ub.upper, 0)
    repn = generate_standard_repn(ub.body)
    self.assertTrue(repn.is_linear())
    simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
    assertExpressionsEqual(self, simplified, m.x1 + m.x2 - m.d + Ms[m.d1][1] * m.d1.binary_indicator_var - Ms[m.d1][1])
    cons = mbm.get_transformed_constraints(m.d2.func)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    ub = cons[1]
    assertExpressionsEqual(self, lb.expr, 0.0 <= 2 * m.x1 + 4 * m.x2 + 7 - m.d - Ms[m.d2][0] * (1 - m.d2.binary_indicator_var))
    self.assertIsNone(ub.lower)
    self.assertEqual(ub.upper, 0)
    repn = generate_standard_repn(ub.body)
    self.assertTrue(repn.is_linear())
    simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
    assertExpressionsEqual(self, simplified, 2 * m.x1 + 4 * m.x2 - m.d + Ms[m.d2][1] * m.d2.binary_indicator_var - (Ms[m.d2][1] - 7))
    cons = mbm.get_transformed_constraints(m.d3.func)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    ub = cons[1]
    assertExpressionsEqual(self, lb.expr, 0.0 <= m.x1 - 5 * m.x2 - 3 - m.d - Ms[m.d3][0] * (1 - m.d3.binary_indicator_var))
    self.assertIsNone(ub.lower)
    self.assertEqual(ub.upper, 0)
    repn = generate_standard_repn(ub.body)
    self.assertTrue(repn.is_linear())
    simplified = repn.constant + sum((repn.linear_coefs[i] * repn.linear_vars[i] for i in range(len(repn.linear_vars))))
    assertExpressionsEqual(self, simplified, m.x1 - 5 * m.x2 - m.d + Ms[m.d3][1] * m.d3.binary_indicator_var - (Ms[m.d3][1] + 3))
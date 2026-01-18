import pickle
from pyomo.common.dependencies import dill
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.base import constraint, ComponentUID
from pyomo.core.base.block import _BlockData
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
from io import StringIO
import random
import pyomo.opt
def check_three_term_xor_constraint(self, transformation):
    m = models.makeThreeTermIndexedDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    xor = m.component('_pyomo_gdp_%s_reformulation' % transformation).component('disjunction_xor')
    self.assertIsInstance(xor, Constraint)
    self.assertEqual(xor[1].lower, 1)
    self.assertEqual(xor[1].upper, 1)
    self.assertEqual(xor[2].lower, 1)
    self.assertEqual(xor[2].upper, 1)
    repn = generate_standard_repn(xor[1].body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 3)
    for i in range(3):
        check_linear_coef(self, repn, m.disjunct[i, 1].indicator_var, 1)
    repn = generate_standard_repn(xor[2].body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 3)
    for i in range(3):
        check_linear_coef(self, repn, m.disjunct[i, 2].indicator_var, 1)
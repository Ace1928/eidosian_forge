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
def check_deactivated_disjunct_leaves_nested_disjunct_active(self, transformation, **kwargs):
    m = models.makeNestedDisjunctions_FlatDisjuncts()
    m.d1.deactivate()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m], **kwargs)
    self.assertFalse(m.d1.active)
    self.assertTrue(m.d1.indicator_var.fixed)
    self.assertEqual(m.d1.indicator_var.value, 0)
    self.assertFalse(m.d2.active)
    self.assertFalse(m.d2.indicator_var.fixed)
    self.assertTrue(m.d3.active)
    self.assertFalse(m.d3.indicator_var.fixed)
    self.assertTrue(m.d4.active)
    self.assertFalse(m.d4.indicator_var.fixed)
    m = models.makeNestedDisjunctions_NestedDisjuncts()
    m.d1.deactivate()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m], **kwargs)
    self.assertFalse(m.d1.active)
    self.assertTrue(m.d1.indicator_var.fixed)
    self.assertEqual(m.d1.indicator_var.value, 0)
    self.assertFalse(m.d2.active)
    self.assertFalse(m.d2.indicator_var.fixed)
    self.assertTrue(m.d1.d3.active)
    self.assertFalse(m.d1.d3.indicator_var.fixed)
    self.assertTrue(m.d1.d4.active)
    self.assertFalse(m.d1.d4.indicator_var.fixed)
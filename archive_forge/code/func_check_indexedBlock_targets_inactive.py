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
def check_indexedBlock_targets_inactive(self, transformation, **kwargs):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.b], **kwargs)
    self.assertTrue(m.disjunct1.active)
    self.assertTrue(m.disjunct1[1, 0].active)
    self.assertTrue(m.disjunct1[1, 1].active)
    self.assertTrue(m.disjunct1[2, 0].active)
    self.assertTrue(m.disjunct1[2, 1].active)
    self.assertIsNone(m.disjunct1[1, 0].transformation_block)
    self.assertIsNone(m.disjunct1[1, 1].transformation_block)
    self.assertIsNone(m.disjunct1[2, 0].transformation_block)
    self.assertIsNone(m.disjunct1[2, 1].transformation_block)
    self.assertFalse(m.b[0].disjunct.active)
    self.assertFalse(m.b[0].disjunct[0].active)
    self.assertFalse(m.b[0].disjunct[1].active)
    self.assertFalse(m.b[1].disjunct0.active)
    self.assertFalse(m.b[1].disjunct1.active)
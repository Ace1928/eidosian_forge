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
def check_indexedDisj_targets_inactive(self, transformation, **kwargs):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.disjunction1], **kwargs)
    self.assertFalse(m.disjunction1.active)
    self.assertFalse(m.disjunction1[1].active)
    self.assertFalse(m.disjunction1[2].active)
    self.assertFalse(m.disjunct1[1, 0].active)
    self.assertFalse(m.disjunct1[1, 1].active)
    self.assertFalse(m.disjunct1[2, 0].active)
    self.assertFalse(m.disjunct1[2, 1].active)
    self.assertFalse(m.disjunct1.active)
    self.assertTrue(m.b[0].disjunct[0].active)
    self.assertTrue(m.b[0].disjunct[1].active)
    self.assertTrue(m.b[1].disjunct0.active)
    self.assertTrue(m.b[1].disjunct1.active)
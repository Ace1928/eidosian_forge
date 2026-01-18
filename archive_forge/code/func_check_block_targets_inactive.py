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
def check_block_targets_inactive(self, transformation, **kwargs):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.b], **kwargs)
    self.assertFalse(m.b.disjunct[0].active)
    self.assertFalse(m.b.disjunct[1].active)
    self.assertFalse(m.b.disjunct.active)
    self.assertTrue(m.simpledisj.active)
    self.assertTrue(m.simpledisj2.active)
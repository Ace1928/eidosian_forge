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
def check_disjuncts_inactive_nested(self, transformation, **kwargs):
    m = models.makeNestedDisjunctions()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,), **kwargs)
    self.assertFalse(m.disjunction.active)
    self.assertFalse(m.simpledisjunct.active)
    self.assertFalse(m.disjunct[0].active)
    self.assertFalse(m.disjunct[1].active)
    self.assertFalse(m.disjunct.active)
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
def check_transform_block_data(self, transformation, **kwargs):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m.b[0], **kwargs)
    self.assertIsNone(m.component('_pyomo_gdp_%s_reformulation' % transformation))
    self.assertIsInstance(m.b[0].component('_pyomo_gdp_%s_reformulation' % transformation), Block)
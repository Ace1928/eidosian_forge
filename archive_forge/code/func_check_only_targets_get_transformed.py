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
def check_only_targets_get_transformed(self, transformation):
    m = models.makeTwoSimpleDisjunctions()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m, targets=[m.disjunction1])
    disjBlock = m.component('_pyomo_gdp_%s_reformulation' % transformation).relaxedDisjuncts
    self.assertEqual(len(disjBlock), 2)
    pairs = [(0, 0), (1, 1)]
    for i, j in pairs:
        self.assertIs(disjBlock[i], m.disjunct1[j].transformation_block)
        self.assertIs(trans.get_src_disjunct(disjBlock[i]), m.disjunct1[j])
    self.assertIsNone(m.disjunct2[0].transformation_block)
    self.assertIsNone(m.disjunct2[1].transformation_block)
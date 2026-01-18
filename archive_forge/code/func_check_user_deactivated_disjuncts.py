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
def check_user_deactivated_disjuncts(self, transformation, check_trans_block=True, **kwargs):
    m = models.makeTwoTermDisj()
    m.d[0].deactivate()
    transform = TransformationFactory('gdp.%s' % transformation)
    transform.apply_to(m, targets=(m,), **kwargs)
    self.assertFalse(m.disjunction.active)
    self.assertFalse(m.d[1].active)
    if check_trans_block:
        rBlock = m.component('_pyomo_gdp_%s_reformulation' % transformation)
        disjBlock = rBlock.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 1)
        self.assertIs(disjBlock[0], m.d[1].transformation_block)
        self.assertIs(transform.get_src_disjunct(disjBlock[0]), m.d[1])
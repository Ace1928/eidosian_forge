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
def checkb0TargetsTransformed(self, m, transformation):
    trans = TransformationFactory('gdp.%s' % transformation)
    disjBlock = m.b[0].component('_pyomo_gdp_%s_reformulation' % transformation).relaxedDisjuncts
    self.assertEqual(len(disjBlock), 2)
    self.assertIs(trans.get_transformed_constraints(m.b[0].disjunct[0].c)[0].parent_block(), disjBlock[0])
    self.assertIs(trans.get_transformed_constraints(m.b[0].disjunct[1].c)[0].parent_block(), disjBlock[1])
    pairs = [(0, 0), (1, 1)]
    for i, j in pairs:
        self.assertIs(m.b[0].disjunct[i].transformation_block, disjBlock[j])
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), m.b[0].disjunct[i])
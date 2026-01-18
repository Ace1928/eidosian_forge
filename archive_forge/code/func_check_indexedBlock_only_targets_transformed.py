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
def check_indexedBlock_only_targets_transformed(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m, targets=[m.b])
    disjBlock1 = m.b[0].component('_pyomo_gdp_%s_reformulation' % transformation).relaxedDisjuncts
    self.assertEqual(len(disjBlock1), 2)
    self.assertIs(trans.get_transformed_constraints(m.b[0].disjunct[0].c)[0].parent_block(), disjBlock1[0])
    self.assertIs(trans.get_transformed_constraints(m.b[0].disjunct[1].c)[0].parent_block(), disjBlock1[1])
    disjBlock2 = m.b[1].component('_pyomo_gdp_%s_reformulation' % transformation).relaxedDisjuncts
    self.assertEqual(len(disjBlock2), 2)
    self.assertIs(trans.get_transformed_constraints(m.b[1].disjunct0.c)[0].parent_block(), disjBlock2[0])
    self.assertIs(trans.get_transformed_constraints(m.b[1].disjunct1.c)[0].parent_block(), disjBlock2[1])
    pairs = {0: [('disjunct', 0, 0), ('disjunct', 1, 1)], 1: [('disjunct0', None, 0), ('disjunct1', None, 1)]}
    for blocknum, lst in pairs.items():
        for comp, i, j in lst:
            original = m.b[blocknum].component(comp)
            if blocknum == 0:
                disjBlock = disjBlock1
            if blocknum == 1:
                disjBlock = disjBlock2
            self.assertIs(original[i].transformation_block, disjBlock[j])
            self.assertIs(trans.get_src_disjunct(disjBlock[j]), original[i])
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
def check_indexedDisj_only_targets_transformed(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m, targets=[m.disjunction1])
    disjBlock = m.component('_pyomo_gdp_%s_reformulation' % transformation).relaxedDisjuncts
    self.assertEqual(len(disjBlock), 4)
    if transformation == 'bigm':
        self.assertIs(trans.get_transformed_constraints(m.disjunct1[1, 0].c)[0].parent_block(), disjBlock[0])
        self.assertIs(trans.get_transformed_constraints(m.disjunct1[1, 1].c)[0].parent_block(), disjBlock[1])
        self.assertIs(trans.get_transformed_constraints(m.disjunct1[2, 0].c)[0].parent_block(), disjBlock[2])
        self.assertIs(trans.get_transformed_constraints(m.disjunct1[2, 1].c)[0].parent_block(), disjBlock[3])
    elif transformation == 'hull':
        self.assertIs(trans.get_transformed_constraints(m.disjunct1[1, 0].c)[0].parent_block().parent_block(), disjBlock[0])
        self.assertIs(trans.get_transformed_constraints(m.disjunct1[1, 1].c)[0].parent_block(), disjBlock[1])
        self.assertIs(trans.get_transformed_constraints(m.disjunct1[2, 0].c)[0].parent_block().parent_block(), disjBlock[2])
        self.assertIs(trans.get_transformed_constraints(m.disjunct1[2, 1].c)[0].parent_block(), disjBlock[3])
    pairs = [((1, 0), 0), ((1, 1), 1), ((2, 0), 2), ((2, 1), 3)]
    for i, j in pairs:
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), m.disjunct1[i])
        self.assertIs(disjBlock[j], m.disjunct1[i].transformation_block)
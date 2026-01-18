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
def check_disjunction_data_target(self, transformation):
    m = models.makeThreeTermIndexedDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.disjunction[2]])
    transBlock = m.component('_pyomo_gdp_%s_reformulation' % transformation)
    self.assertIsInstance(transBlock, Block)
    self.assertIsInstance(transBlock.component('disjunction_xor'), Constraint)
    self.assertIsInstance(transBlock.disjunction_xor[2], constraint._GeneralConstraintData)
    self.assertIsInstance(transBlock.component('relaxedDisjuncts'), Block)
    self.assertEqual(len(transBlock.relaxedDisjuncts), 3)
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.disjunction[1]])
    self.assertIsInstance(m.disjunction[1].algebraic_constraint, constraint._GeneralConstraintData)
    transBlock = m.component('_pyomo_gdp_%s_reformulation_4' % transformation)
    self.assertIsInstance(transBlock, Block)
    self.assertEqual(len(transBlock.relaxedDisjuncts), 3)
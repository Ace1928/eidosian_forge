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
def check_silly_target(self, transformation, **kwargs):
    m = models.makeTwoTermDisj()
    self.assertRaisesRegex(GDP_Error, "Target 'd\\[1\\].c1' was not a Block, Disjunct, or Disjunction. It was of type <class 'pyomo.core.base.constraint.ScalarConstraint'> and can't be transformed.", TransformationFactory('gdp.%s' % transformation).apply_to, m, targets=[m.d[1].c1], **kwargs)
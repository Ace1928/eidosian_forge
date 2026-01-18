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
def check_indicator_vars(self, transformation):
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    oldblock = m.component('d')
    _binary0 = oldblock[0].binary_indicator_var
    self.assertIsInstance(_binary0, Var)
    self.assertTrue(_binary0.active)
    self.assertTrue(_binary0.is_binary())
    _binary1 = oldblock[1].binary_indicator_var
    self.assertIsInstance(_binary1, Var)
    self.assertTrue(_binary1.active)
    self.assertTrue(_binary1.is_binary())
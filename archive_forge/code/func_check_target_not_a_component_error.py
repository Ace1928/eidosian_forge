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
def check_target_not_a_component_error(self, transformation, **kwargs):
    decoy = ConcreteModel()
    decoy.block = Block()
    m = models.makeTwoSimpleDisjunctions()
    self.assertRaisesRegex(GDP_Error, "Target 'block' is not a component on instance 'unknown'!", TransformationFactory('gdp.%s' % transformation).apply_to, m, targets=[decoy.block], **kwargs)
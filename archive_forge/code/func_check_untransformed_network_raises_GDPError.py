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
def check_untransformed_network_raises_GDPError(self, transformation, **kwargs):
    m = models.makeNetworkDisjunction()
    self.assertRaisesRegex(GDP_Error, "No %s transformation handler registered for modeling components of type <class 'pyomo.network.arc.Arc'>. If your disjuncts contain non-GDP Pyomo components that require transformation, please transform them first." % transformation, TransformationFactory('gdp.%s' % transformation).apply_to, m, **kwargs)
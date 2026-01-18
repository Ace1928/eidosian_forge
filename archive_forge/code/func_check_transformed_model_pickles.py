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
def check_transformed_model_pickles(self, transformation):
    m = models.makeLogicalConstraintsOnDisjuncts_NonlinearConvex()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)
    unpickle = pickle.loads(pickle.dumps(m))
    check_pprint_equal(self, m, unpickle)
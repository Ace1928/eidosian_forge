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
def check_ask_for_transformed_constraint_from_untransformed_disjunct(self, transformation):
    m = models.makeTwoTermIndexedDisjunction()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m, targets=m.disjunction[1])
    self.assertRaisesRegex(GDP_Error, "Constraint 'disjunct\\[2,b\\].cons_b' is on a disjunct which has not been transformed", trans.get_transformed_constraints, m.disjunct[2, 'b'].cons_b)
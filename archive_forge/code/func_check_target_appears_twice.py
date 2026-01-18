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
def check_target_appears_twice(self, transformation):
    m = models.makeNestedDisjunctions_NestedDisjuncts()
    m1 = TransformationFactory('gdp.%s' % transformation).create_using(m, targets=[m.d1, m.disj])
    check_all_components_transformed(self, m1)
    check_transformation_blocks_nestedDisjunctions(self, m1, transformation)
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.disj, m.d1])
    check_all_components_transformed(self, m)
    check_transformation_blocks_nestedDisjunctions(self, m, transformation)
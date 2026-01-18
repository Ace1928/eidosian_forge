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
def check_partial_deactivate_indexed_disjunction(self, transformation):
    """Test for partial deactivation of an indexed disjunction."""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))

    @m.Disjunction([0, 1])
    def disj(m, i):
        if i == 0:
            return [m.x >= 1, m.x >= 2]
        else:
            return [m.x >= 3, m.x >= 4]
    m.disj[0].disjuncts[0].indicator_var.fix(1)
    m.disj[0].disjuncts[1].indicator_var.fix(1)
    m.disj[0].deactivate()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    transBlock = m.component('_pyomo_gdp_%s_reformulation' % transformation)
    self.assertEqual(len(transBlock.disj_xor), 1, 'There should only be one XOR constraint generated. Found %s.' % len(transBlock.disj_xor))
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
def check_warn_for_untransformed(self, transformation, **kwargs):
    m = models.makeDisjunctionsOnIndexedBlock()

    def innerdisj_rule(d, flag):
        m = d.model()
        if flag:
            d.c = Constraint(expr=m.a[1] <= 2)
        else:
            d.c = Constraint(expr=m.a[1] >= 65)
    m.disjunct1[1, 1].innerdisjunct = Disjunct([0, 1], rule=innerdisj_rule)
    m.disjunct1[1, 1].innerdisjunction = Disjunction([0], rule=lambda a, i: [m.disjunct1[1, 1].innerdisjunct[0], m.disjunct1[1, 1].innerdisjunct[1]])
    m.disjunct1[1, 1].innerdisjunction.deactivate()
    self.assertRaisesRegex(GDP_Error, "Found active disjunct 'disjunct1\\[1,1\\].innerdisjunct\\[0\\]' in disjunct 'disjunct1\\[1,1\\]'!.*", TransformationFactory('gdp.%s' % transformation).create_using, m, targets=[m.disjunction1[1]], **kwargs)
    m.disjunct1[1, 1].innerdisjunction.activate()
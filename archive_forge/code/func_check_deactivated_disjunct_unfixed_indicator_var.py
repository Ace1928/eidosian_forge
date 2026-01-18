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
def check_deactivated_disjunct_unfixed_indicator_var(self, transformation, **kwargs):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 8))
    m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])
    m.disjunction.disjuncts[0].deactivate()
    m.disjunction.disjuncts[0].indicator_var.fixed = False
    self.assertRaisesRegex(GDP_Error, "The disjunct 'disjunction_disjuncts\\[0\\]' is deactivated, but the indicator_var is not fixed and the disjunct does not appear to have been transformed. This makes no sense. \\(If the intent is to deactivate the disjunct, fix its indicator_var to False.\\)", TransformationFactory('gdp.%s' % transformation).apply_to, m, **kwargs)
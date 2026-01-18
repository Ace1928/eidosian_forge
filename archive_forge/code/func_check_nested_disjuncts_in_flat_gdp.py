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
def check_nested_disjuncts_in_flat_gdp(self, transformation):
    m = models.make_non_nested_model_declaring_Disjuncts_on_each_other()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    SolverFactory('gurobi').solve(m)
    self.assertAlmostEqual(value(m.obj), 1020)
    for t in m.T:
        self.assertTrue(value(m.disj1[t].indicator_var))
        self.assertTrue(value(m.disj1[t].sub1.indicator_var))
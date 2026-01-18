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
def check_squared_term_coef(self, repn, var, coef):
    var_id = None
    for i, (v1, v2) in enumerate(repn.quadratic_vars):
        if v1 is var and v2 is var:
            var_id = i
            break
    self.assertIsNotNone(var_id)
    self.assertEqual(repn.quadratic_coefs[var_id], coef)
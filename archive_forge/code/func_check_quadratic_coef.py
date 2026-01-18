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
def check_quadratic_coef(self, repn, v1, v2, coef):
    if isinstance(v1, BooleanVar):
        v1 = v1.get_associated_binary()
    if isinstance(v2, BooleanVar):
        v2 = v2.get_associated_binary()
    v1id = id(v1)
    v2id = id(v2)
    qcoef_map = dict()
    for (_v1, _v2), _coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
        qcoef_map[id(_v1), id(_v2)] = _coef
        qcoef_map[id(_v2), id(_v1)] = _coef
    self.assertIn((v1id, v2id), qcoef_map)
    self.assertAlmostEqual(qcoef_map[v1id, v2id], coef)
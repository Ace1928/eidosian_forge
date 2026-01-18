import pickle
import os
import io
import sys
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.relational_expr import (
def test_ineq(self):
    M = ConcreteModel()
    M.v = Var()
    e = M.v >= 0
    s = pickle.dumps(e)
    e_ = pickle.loads(s)
    self.assertEqual(str(e), str(e_))
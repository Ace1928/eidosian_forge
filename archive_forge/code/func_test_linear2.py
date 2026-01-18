import pickle
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types, as_numeric, value
from pyomo.core.expr.visitor import replace_expressions
from pyomo.repn import generate_standard_repn
from pyomo.environ import (
import pyomo.kernel
def test_linear2(self):
    m = ConcreteModel()
    m.A = RangeSet(5)
    m.v = Var(m.A, initialize=1)
    m.p = Param(m.A, initialize={1: -2, 2: -1, 3: 0, 4: 1, 5: 2})
    e = quicksum((m.p[i] * m.v[1] for i in m.p)) + summation(m.p, m.v)
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '-2*v[1] - v[2] + v[4] + 2*v[5]')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), '-2*v[1] - v[2] + v[4] + 2*v[5]')
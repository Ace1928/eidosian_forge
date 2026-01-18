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
def test_external_fn(self):

    def _g(*args):
        return len(args)
    m = ConcreteModel()
    m.v = Var(initialize=1)
    m.v.fixed = True
    m.g = ExternalFunction(_g)
    e = 100 * m.g(1, 2.0, '3')
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '300')
    self.assertEqual(rep.polynomial_degree(), 0)
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(rep.polynomial_degree(), 0)
    e = 100 * m.g(1, 2.0, '3', m.v)
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '400')
    self.assertEqual(rep.polynomial_degree(), 0)
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(rep.polynomial_degree(), None)
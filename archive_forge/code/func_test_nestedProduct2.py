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
def test_nestedProduct2(self):
    m = ConcreteModel()
    m.a = Param(default=2)
    m.b = Param(default=3)
    m.c = Param(default=5)
    m.d = Var()
    e1 = m.a + m.b
    e2 = m.c + e1
    e3 = e1 + m.d
    e = e2 * e3
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 1)
    self.assertTrue(len(rep.linear_coefs) == 1)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {None: 50, id(m.d): 10}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {None: 50, id(rep.linear_vars[0]): 10}
    self.assertEqual(baseline, repn_to_dict(rep))
    e1 = m.a + m.b
    e2 = m.c * e1
    e3 = e1 * m.d
    e = e2 * e3
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 1)
    self.assertTrue(len(rep.linear_coefs) == 1)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.d): 125}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {id(rep.linear_vars[0]): 125}
    self.assertEqual(baseline, repn_to_dict(rep))
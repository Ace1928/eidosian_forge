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
def test_expr_const1(self):
    o = pyomo.kernel.expression()
    o.expr = as_numeric(2)
    e = 1000 * o
    rep = generate_standard_repn(e, compute_values=True)
    self.assertTrue(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 0)
    self.assertTrue(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {None: 2000}
    self.assertEqual(baseline, repn_to_dict(rep))
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
def test_linear_sum3(self):
    m = ConcreteModel()
    m.A = Set(initialize=range(5))
    m.x = Var(m.A, initialize=3)
    m.p = Param(m.A, mutable=True, default=1)
    e = quicksum(((i + 1) * m.x[i] for i in m.A))
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 5)
    self.assertTrue(len(rep.linear_coefs) == 5)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.x[0]): 1, id(m.x[1]): 2, id(m.x[2]): 3, id(m.x[3]): 4, id(m.x[4]): 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    m.x[2].fixed = True
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 4)
    self.assertTrue(len(rep.linear_coefs) == 4)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.x[0]): 1, id(m.x[1]): 2, None: 9, id(m.x[3]): 4, id(m.x[4]): 5}
    self.assertEqual(baseline, repn_to_dict(rep))
import os
from os.path import abspath, dirname
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet
from pyomo.core import (
from pyomo.core.base import TransformationFactory
from pyomo.core.expr import log
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.gdp import Disjunction, Disjunct
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.opt import SolverFactory, check_available_solvers
import pyomo.contrib.fme.fourier_motzkin_elimination
from io import StringIO
import logging
import random
def test_integer_arithmetic_non1_coefficients(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 9))
    m.y = Var(bounds=(-5, 5))
    m.c1 = Constraint(expr=4 * m.x + m.y >= 4)
    m.c2 = Constraint(expr=m.y >= 2 * m.x)
    fme = TransformationFactory('contrib.fourier_motzkin_elimination')
    fme.apply_to(m, vars_to_eliminate=m.x, constraint_filtering_callback=None, do_integer_arithmetic=True, verbose=True)
    constraints = m._pyomo_contrib_fme_transformation.projected_constraints
    self.assertEqual(len(constraints), 3)
    cons = constraints[3]
    self.assertEqual(value(cons.lower), -32)
    self.assertIs(cons.body, m.y)
    self.assertIsNone(cons.upper)
    cons = constraints[2]
    self.assertEqual(value(cons.lower), 0)
    self.assertIsNone(cons.upper)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_coefs), 1)
    self.assertIs(repn.linear_vars[0], m.y)
    self.assertEqual(repn.linear_coefs[0], 2)
    cons = constraints[1]
    self.assertEqual(value(cons.lower), 4)
    self.assertIsNone(cons.upper)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_coefs), 1)
    self.assertIs(repn.linear_vars[0], m.y)
    self.assertEqual(repn.linear_coefs[0], 3)
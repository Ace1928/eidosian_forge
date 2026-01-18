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
def check_projected_constraints(self, m, indices):
    constraints = m._pyomo_contrib_fme_transformation.projected_constraints
    cons = constraints[indices[0]]
    self.assertEqual(value(cons.lower), -1)
    self.assertIsNone(cons.upper)
    body = generate_standard_repn(cons.body)
    self.assertTrue(body.is_linear())
    linear_vars = body.linear_vars
    coefs = body.linear_coefs
    self.assertEqual(len(linear_vars), 2)
    self.assertIs(linear_vars[0], m.x)
    self.assertEqual(coefs[0], -1)
    self.assertIs(linear_vars[1], m.y)
    self.assertEqual(coefs[1], 0.01)
    cons = constraints[indices[1]]
    self.assertEqual(value(cons.lower), -1000)
    self.assertIsNone(cons.upper)
    body = generate_standard_repn(cons.body)
    linear_vars = body.linear_vars
    coefs = body.linear_coefs
    self.assertEqual(len(linear_vars), 2)
    self.assertIs(linear_vars[0], m.u[1])
    self.assertEqual(coefs[0], -1000)
    self.assertIs(linear_vars[1], m.y)
    self.assertEqual(coefs[1], -1)
    cons = constraints[indices[2]]
    self.assertEqual(value(cons.lower), -999)
    self.assertIsNone(cons.upper)
    body = generate_standard_repn(cons.body)
    linear_vars = body.linear_vars
    coefs = body.linear_coefs
    self.assertEqual(len(linear_vars), 3)
    self.assertIs(linear_vars[0], m.u[2])
    self.assertEqual(coefs[0], -1000)
    self.assertIs(linear_vars[1], m.x)
    self.assertEqual(coefs[1], 1)
    self.assertIs(linear_vars[2], m.y)
    self.assertEqual(coefs[2], -0.01)
    cons = constraints[indices[3]]
    self.assertEqual(value(cons.lower), 1)
    self.assertIsNone(cons.upper)
    body = generate_standard_repn(cons.body)
    linear_vars = body.linear_vars
    coefs = body.linear_coefs
    self.assertEqual(len(linear_vars), 2)
    self.assertIs(linear_vars[1], m.u[2])
    self.assertEqual(coefs[1], 1)
    self.assertIs(linear_vars[0], m.u[1])
    self.assertEqual(coefs[0], 100)
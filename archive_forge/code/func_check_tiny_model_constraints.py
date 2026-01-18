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
def check_tiny_model_constraints(self, constraints):
    m = constraints.model()
    self.assertEqual(len(constraints), 1)
    cons = constraints[1]
    self.assertEqual(value(cons.lower), -5)
    self.assertIsNone(cons.upper)
    repn = generate_standard_repn(cons.body)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertIs(repn.linear_vars[0], m.x)
    self.assertEqual(repn.linear_coefs[0], -1)
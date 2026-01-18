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
def test_combine_three_inequalities_and_flatten_blocks(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.b = Block()
    m.b.c = Constraint(expr=m.x >= 2)
    m.c = Constraint(expr=m.y <= m.x)
    m.b.b2 = Block()
    m.b.b2.c = Constraint(expr=m.y >= 4)
    TransformationFactory('contrib.fourier_motzkin_elimination').apply_to(m, vars_to_eliminate=m.y, do_integer_arithmetic=True)
    constraints = m._pyomo_contrib_fme_transformation.projected_constraints
    self.assertEqual(len(constraints), 2)
    cons = constraints[1]
    self.assertEqual(value(cons.lower), 2)
    self.assertIsNone(cons.upper)
    self.assertIs(cons.body, m.x)
    cons = constraints[2]
    self.assertEqual(value(cons.lower), 4)
    self.assertIsNone(cons.upper)
    self.assertIs(cons.body, m.x)
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
def test_infeasible_model(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))
    m.cons1 = Constraint(expr=m.x >= 6)
    m.cons2 = Constraint(expr=m.x <= 2)
    self.assertRaisesRegex(RuntimeError, 'Fourier-Motzkin found the model is infeasible!', TransformationFactory('contrib.fourier_motzkin_elimination').apply_to, m, vars_to_eliminate=m.x)
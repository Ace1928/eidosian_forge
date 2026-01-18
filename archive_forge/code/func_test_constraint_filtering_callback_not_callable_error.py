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
def test_constraint_filtering_callback_not_callable_error(self):
    m = self.makeModel()
    fme = TransformationFactory('contrib.fourier_motzkin_elimination')
    log = StringIO()
    with LoggingIntercept(log, 'pyomo.contrib.fme', logging.ERROR):
        self.assertRaisesRegex(TypeError, "'int' object is not callable", fme.apply_to, m, vars_to_eliminate=m.x, constraint_filtering_callback=5)
    self.assertRegex(log.getvalue(), 'Problem calling constraint filter callback on constraint with right-hand side -1.0 and body:*')
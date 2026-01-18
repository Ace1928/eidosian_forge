from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
def check_furman_et_al_denominator(self, expr, ind_var):
    self.assertEqual(expr._const, EPS)
    self.assertEqual(len(expr._args), 1)
    self.assertEqual(len(expr._coef), 1)
    self.assertEqual(expr._coef[0], 1 - EPS)
    self.assertIs(expr._args[0], ind_var)
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
@d.Disjunct()
def d_r(e):
    e.lambdas = Var(m.S, bounds=(0, 1))
    e.LocalVars = Suffix(direction=Suffix.LOCAL)
    e.LocalVars[e] = list(e.lambdas.values())
    e.c1 = Constraint(expr=e.lambdas[1] + e.lambdas[2] == 1)
    e.c2 = Constraint(expr=m.x == 2 * e.lambdas[1] + 3 * e.lambdas[2])
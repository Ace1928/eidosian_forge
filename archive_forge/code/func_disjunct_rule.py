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
def disjunct_rule(d, i):
    m = d.model()
    if i:
        d.cons_block = Constraint(expr=m.b.x >= 5)
        d.cons_model = Constraint(expr=m.component('b.x') == 0)
    else:
        d.cons_model = Constraint(expr=m.component('b.x') <= -5)
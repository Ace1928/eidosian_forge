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
def test_hull_chull_equivalent(self):
    m = models.makeTwoTermDisj()
    out1 = StringIO()
    out2 = StringIO()
    m1 = TransformationFactory('gdp.hull').create_using(m)
    m2 = TransformationFactory('gdp.chull').create_using(m)
    m1.pprint(ostream=out1)
    m2.pprint(ostream=out2)
    self.assertMultiLineEqual(out1.getvalue(), out2.getvalue())
from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
def test_arg_M_simpleConstraint(self):
    m = models.makeTwoTermDisj()
    m.BigM = Suffix(direction=Suffix.LOCAL)
    m.BigM[None] = 20
    m.BigM[m.d[0].c] = 200
    m.BigM[m.d[1].c1] = 200
    m.BigM[m.d[1].c2] = 200
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(m, bigM={None: 19, m.d[0].c: 18, m.d[1].c1: 17, m.d[1].c2: 16})
    self.checkMs(m, bigm, -18, -17, 17, 16)
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
def test_arg_M_indexedConstraint(self):
    m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
    m.BigM = Suffix(direction=Suffix.LOCAL)
    m.BigM[None] = 20
    m.BigM[m.disjunct[0].c] = 19
    TransformationFactory('gdp.bigm').apply_to(m, bigM=ComponentMap({None: 19, m.disjunct[0].c: 17}))
    self.checkMs(m, -17, -17, -19, 19, -19, 19)
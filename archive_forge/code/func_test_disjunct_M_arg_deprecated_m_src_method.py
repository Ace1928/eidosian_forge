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
def test_disjunct_M_arg_deprecated_m_src_method(self):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    bigm = TransformationFactory('gdp.bigm')
    bigms = {m.b: 100, m.b.disjunct[1]: 13}
    bigm.apply_to(m, bigM=bigms)
    self.checkMs(m, -100, 100, 13, -3, 1.5)
    src, key = bigm.get_m_value_src(m.simpledisj.c)
    self.assertEqual(src, -3)
    self.assertIsNone(key)
    src, key = bigm.get_m_value_src(m.simpledisj2.c)
    self.assertIsNone(src)
    self.assertEqual(key, 1.5)
    src, key = bigm.get_m_value_src(m.b.disjunct[0].c)
    self.assertIs(src, bigms)
    self.assertIs(key, m.b)
    src, key = bigm.get_m_value_src(m.b.disjunct[1].c)
    self.assertIs(src, bigms)
    self.assertIs(key, m.b.disjunct[1])
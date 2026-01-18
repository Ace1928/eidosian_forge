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
def test_m_value_mappings(self):
    m = models.makeNestedDisjunctions()
    bigm = TransformationFactory('gdp.bigm')
    m.simpledisjunct.BigM = Suffix(direction=Suffix.LOCAL)
    m.simpledisjunct.BigM[None] = 58
    m.simpledisjunct.BigM[m.simpledisjunct.innerdisjunct0.c] = 42
    bigms = {m.disjunct[1].innerdisjunct[0]: 89}
    bigm.apply_to(m, bigM=bigms)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.disjunct[1].innerdisjunct[0].c)
    self.assertIs(l_src, bigms)
    self.assertIs(u_src, bigms)
    self.assertIs(l_key, m.disjunct[1].innerdisjunct[0])
    self.assertIs(u_key, m.disjunct[1].innerdisjunct[0])
    self.assertEqual(l_val, -89)
    self.assertEqual(u_val, 89)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.disjunct[1].innerdisjunct[1].c)
    self.assertIsNone(l_src)
    self.assertIsNone(u_src)
    self.assertIsNone(l_key)
    self.assertIsNone(u_key)
    self.assertEqual(l_val, -5)
    self.assertIsNone(u_val)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.disjunct[0].c)
    self.assertIsNone(l_src)
    self.assertIsNone(u_src)
    self.assertIsNone(l_key)
    self.assertIsNone(u_key)
    self.assertEqual(l_val, -11)
    self.assertEqual(u_val, 7)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.disjunct[1].c)
    self.assertIsNone(l_src)
    self.assertIsNone(u_src)
    self.assertIsNone(l_key)
    self.assertIsNone(u_key)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 21)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.simpledisjunct.innerdisjunct0.c)
    self.assertIsNone(l_src)
    self.assertIs(u_src, m.simpledisjunct.BigM)
    self.assertIsNone(l_key)
    self.assertIs(u_key, m.simpledisjunct.innerdisjunct0.c)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 42)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.simpledisjunct.innerdisjunct1.c)
    self.assertIs(l_src, m.simpledisjunct.BigM)
    self.assertIsNone(u_src)
    self.assertIsNone(l_key)
    self.assertIsNone(u_key)
    self.assertEqual(l_val, -58)
    self.assertIsNone(u_val)
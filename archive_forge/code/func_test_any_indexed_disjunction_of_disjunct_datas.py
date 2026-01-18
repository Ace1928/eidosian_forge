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
def test_any_indexed_disjunction_of_disjunct_datas(self):
    m = models.makeAnyIndexedDisjunctionOfDisjunctDatas()
    TransformationFactory('gdp.hull').apply_to(m)
    self.check_trans_block_disjunctions_of_disjunct_datas(m)
    transBlock = m.component('_pyomo_gdp_hull_reformulation')
    self.assertIsInstance(transBlock.component('disjunction_xor'), Constraint)
    self.assertEqual(len(transBlock.component('disjunction_xor')), 2)
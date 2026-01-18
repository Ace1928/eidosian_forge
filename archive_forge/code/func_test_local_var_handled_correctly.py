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
def test_local_var_handled_correctly(self):
    m = self.makeModel()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    self.assertIs(hull.get_disaggregated_var(m.x, m.disj1), m.x)
    self.assertEqual(m.x.lb, 0)
    self.assertEqual(m.x.ub, 5)
    self.assertIsNone(m.disj1.transformation_block.disaggregatedVars.component('x'))
    self.assertIsInstance(m.disj1.transformation_block.disaggregatedVars.component('y'), Var)
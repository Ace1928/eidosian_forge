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
def test_locally_declared_variables_disaggregated(self):
    m = models.localVar()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    disj1y = hull.get_disaggregated_var(m.disj2.y, m.disj1)
    disj2y = hull.get_disaggregated_var(m.disj2.y, m.disj2)
    self.assertIs(disj1y, m.disj1.transformation_block.parent_block()._disaggregatedVars[0])
    self.assertIs(disj2y, m.disj2.transformation_block.disaggregatedVars.component('disj2.y'))
    self.assertIs(hull.get_src_var(disj1y), m.disj2.y)
    self.assertIs(hull.get_src_var(disj2y), m.disj2.y)
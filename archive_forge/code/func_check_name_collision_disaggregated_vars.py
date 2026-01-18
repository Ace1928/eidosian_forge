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
def check_name_collision_disaggregated_vars(self, m, disj):
    hull = TransformationFactory('gdp.hull')
    transBlock = disj.transformation_block
    varBlock = transBlock.disaggregatedVars
    self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 2)
    x2 = varBlock.component("'disj1.x'")
    x = varBlock.component('disj1.x')
    x_orig = m.component('disj1.x')
    self.assertIsInstance(x, Var)
    self.assertIsInstance(x2, Var)
    self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
    self.assertIs(hull.get_src_var(x), m.disj1.x)
    self.assertIs(hull.get_disaggregated_var(x_orig, disj), x2)
    self.assertIs(hull.get_src_var(x2), x_orig)
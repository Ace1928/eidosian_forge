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
def test_disaggregation_constraints(self):
    m = self.makeModel()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    disaggregationConstraints = m._pyomo_gdp_hull_reformulation.disaggregationConstraints
    consmap = [(m.component('b.x'), disaggregationConstraints[0]), (m.b.x, disaggregationConstraints[1])]
    for v, cons in consmap:
        disCons = hull.get_disaggregation_constraint(v, m.disjunction)
        self.assertIs(disCons, cons)
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
def test_disaggregation_constraint(self):
    m = models.makeTwoTermDisj_Nonlinear()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    transBlock = m._pyomo_gdp_hull_reformulation
    disjBlock = transBlock.relaxedDisjuncts
    self.check_disaggregation_constraint(hull.get_disaggregation_constraint(m.w, m.disjunction), m.w, transBlock._disaggregatedVars[1], disjBlock[1].disaggregatedVars.w)
    self.check_disaggregation_constraint(hull.get_disaggregation_constraint(m.x, m.disjunction), m.x, disjBlock[0].disaggregatedVars.x, disjBlock[1].disaggregatedVars.x)
    self.check_disaggregation_constraint(hull.get_disaggregation_constraint(m.y, m.disjunction), m.y, transBlock._disaggregatedVars[0], disjBlock[0].disaggregatedVars.y)
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
def test_disaggregatedVar_bounds(self):
    m = models.makeTwoTermDisj_Nonlinear()
    TransformationFactory('gdp.hull').apply_to(m)
    transBlock = m._pyomo_gdp_hull_reformulation
    disjBlock = transBlock.relaxedDisjuncts
    for i in [0, 1]:
        self.check_bound_constraints_on_disjBlock(disjBlock[i].x_bounds, disjBlock[i].disaggregatedVars.x, m.d[i].indicator_var, 1, 8)
        if i == 1:
            self.check_bound_constraints_on_disjBlock(disjBlock[i].w_bounds, disjBlock[i].disaggregatedVars.w, m.d[i].indicator_var, 2, 7)
            self.check_bound_constraints_on_disjunctionBlock(transBlock._boundsConstraints[0, 'lb'], transBlock._boundsConstraints[0, 'ub'], transBlock._disaggregatedVars[0], m.d[0].indicator_var, -10, -3)
        elif i == 0:
            self.check_bound_constraints_on_disjBlock(disjBlock[i].y_bounds, disjBlock[i].disaggregatedVars.y, m.d[i].indicator_var, -10, -3)
            self.check_bound_constraints_on_disjunctionBlock(transBlock._boundsConstraints[1, 'lb'], transBlock._boundsConstraints[1, 'ub'], transBlock._disaggregatedVars[1], m.d[1].indicator_var, 2, 7)
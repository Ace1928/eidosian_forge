import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var
def check_disjunct(self, d, not_pts):
    self.assertEqual(len(d.component_map(Constraint)), 1)
    self.assertEqual(len(d.component_map(Var)), 1)
    self.assertIsInstance(d.lambdas_zero_for_other_simplices, Constraint)
    self.assertEqual(len(d.lambdas_zero_for_other_simplices), len(not_pts))
    transBlock = d.parent_block()
    for i, cons in zip(not_pts, d.lambdas_zero_for_other_simplices.values()):
        assertExpressionsEqual(self, cons.expr, transBlock.lambdas[i] <= 0)
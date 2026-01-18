import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var
def check_log_trans_block_structure(self, transBlock):
    m = transBlock.model()
    self.assertEqual(len(transBlock.component_map(Disjunct)), 1)
    self.assertEqual(len(transBlock.component_map(Disjunction)), 1)
    self.assertEqual(len(transBlock.component_map(Var)), 2)
    self.assertIsInstance(transBlock.substitute_var, Var)
    self.assertIsInstance(transBlock.lambdas, Var)
    self.assertEqual(len(transBlock.lambdas), 4)
    for lamb in transBlock.lambdas.values():
        self.assertEqual(lamb.lb, 0)
        self.assertEqual(lamb.ub, 1)
    self.assertIsInstance(transBlock.convex_combo, Constraint)
    assertExpressionsEqual(self, transBlock.convex_combo.expr, transBlock.lambdas[0] + transBlock.lambdas[1] + transBlock.lambdas[2] + transBlock.lambdas[3] == 1)
    self.assertIsInstance(transBlock.linear_combo, Constraint)
    self.assertEqual(len(transBlock.linear_combo), 1)
    pts = m.pw_log._points
    assertExpressionsEqual(self, transBlock.linear_combo[0].expr, m.x == pts[0][0] * transBlock.lambdas[0] + pts[1][0] * transBlock.lambdas[1] + pts[2][0] * transBlock.lambdas[2] + pts[3][0] * transBlock.lambdas[3])
    self.assertIsInstance(transBlock.linear_func, Constraint)
    self.assertEqual(len(transBlock.linear_func), 1)
    assertExpressionsEqual(self, transBlock.linear_func.expr, transBlock.lambdas[0] * m.f1(1) + transBlock.lambdas[1] * m.f1(3) + transBlock.lambdas[2] * m.f2(6) + transBlock.lambdas[3] * m.f3(10) == transBlock.substitute_var, places=7)
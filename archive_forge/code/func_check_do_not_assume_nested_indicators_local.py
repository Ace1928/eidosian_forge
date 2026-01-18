import pickle
from pyomo.common.dependencies import dill
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.base import constraint, ComponentUID
from pyomo.core.base.block import _BlockData
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
from io import StringIO
import random
import pyomo.opt
def check_do_not_assume_nested_indicators_local(self, transformation):
    m = models.why_indicator_vars_are_not_always_local()
    TransformationFactory(transformation).apply_to(m)
    results = SolverFactory('gurobi').solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(value(m.obj), 9)
    self.assertAlmostEqual(value(m.x), 9)
    self.assertTrue(value(m.Y2.indicator_var))
    self.assertFalse(value(m.Y1.indicator_var))
    self.assertTrue(value(m.Z1.indicator_var))
    self.assertTrue(value(m.Z1.indicator_var))
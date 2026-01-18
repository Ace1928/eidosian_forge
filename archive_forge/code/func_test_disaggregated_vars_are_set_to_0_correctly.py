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
@unittest.skipIf(not linear_solvers, 'No linear solver available')
def test_disaggregated_vars_are_set_to_0_correctly(self):
    m = models.makeNestedDisjunctions_FlatDisjuncts()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    m.d1.indicator_var.fix(False)
    m.d2.indicator_var.fix(True)
    m.d3.indicator_var.fix(False)
    m.d4.indicator_var.fix(False)
    results = SolverFactory(linear_solvers[0]).solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertEqual(value(m.x), 1.1)
    self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d1)), 0)
    self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d2)), 1.1)
    self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d3)), 0)
    self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d4)), 0)
    m.d1.indicator_var.fix(True)
    m.d2.indicator_var.fix(False)
    m.d3.indicator_var.fix(True)
    m.d4.indicator_var.fix(False)
    results = SolverFactory(linear_solvers[0]).solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertEqual(value(m.x), 1.2)
    self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d1)), 1.2)
    self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d2)), 0)
    self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d3)), 1.2)
    self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d4)), 0)
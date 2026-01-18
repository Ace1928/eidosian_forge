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
def test_global_vars_local_to_a_disjunction_disaggregated(self):
    m = ConcreteModel()
    m.disj1 = Disjunct()
    m.disj1.x = Var(bounds=(1, 10))
    m.disj1.y = Var(bounds=(2, 11))
    m.disj1.cons1 = Constraint(expr=m.disj1.x + m.disj1.y <= 5)
    m.disj2 = Disjunct()
    m.disj2.cons = Constraint(expr=m.disj1.y >= 8)
    m.disjunction1 = Disjunction(expr=[m.disj1, m.disj2])
    m.disj3 = Disjunct()
    m.disj3.cons = Constraint(expr=m.disj1.x >= 7)
    m.disj4 = Disjunct()
    m.disj4.cons = Constraint(expr=m.disj1.y == 3)
    m.disjunction2 = Disjunction(expr=[m.disj3, m.disj4])
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    disj = m.disj1
    transBlock = disj.transformation_block
    varBlock = transBlock.disaggregatedVars
    self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 2)
    x = varBlock.component('disj1.x')
    y = varBlock.component('disj1.y')
    self.assertIsInstance(x, Var)
    self.assertIsInstance(y, Var)
    self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
    self.assertIs(hull.get_src_var(x), m.disj1.x)
    self.assertIs(hull.get_disaggregated_var(m.disj1.y, disj), y)
    self.assertIs(hull.get_src_var(y), m.disj1.y)
    for disj in [m.disj2, m.disj4]:
        transBlock = disj.transformation_block
        varBlock = transBlock.disaggregatedVars
        self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 1)
        y = varBlock.component('disj1.y')
        self.assertIsInstance(y, Var)
        self.assertIs(hull.get_disaggregated_var(m.disj1.y, disj), y)
        self.assertIs(hull.get_src_var(y), m.disj1.y)
    disj = m.disj3
    transBlock = disj.transformation_block
    varBlock = transBlock.disaggregatedVars
    self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 1)
    x = varBlock.component('disj1.x')
    self.assertIsInstance(x, Var)
    self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
    self.assertIs(hull.get_src_var(x), m.disj1.x)
    x2 = m.disjunction1.algebraic_constraint.parent_block()._disaggregatedVars[0]
    self.assertIs(hull.get_disaggregated_var(m.disj1.x, m.disj2), x2)
    self.assertIs(hull.get_src_var(x2), m.disj1.x)
    agg_cons = hull.get_disaggregation_constraint(m.disj1.x, m.disjunction1)
    assertExpressionsEqual(self, agg_cons.expr, m.disj1.x == x2 + hull.get_disaggregated_var(m.disj1.x, m.disj1))
    x2 = m.disjunction2.algebraic_constraint.parent_block()._disaggregatedVars[1]
    y1 = m.disjunction2.algebraic_constraint.parent_block()._disaggregatedVars[2]
    self.assertIs(hull.get_disaggregated_var(m.disj1.x, m.disj4), x2)
    self.assertIs(hull.get_src_var(x2), m.disj1.x)
    self.assertIs(hull.get_disaggregated_var(m.disj1.y, m.disj3), y1)
    self.assertIs(hull.get_src_var(y1), m.disj1.y)
    agg_cons = hull.get_disaggregation_constraint(m.disj1.x, m.disjunction2)
    assertExpressionsEqual(self, agg_cons.expr, m.disj1.x == x2 + hull.get_disaggregated_var(m.disj1.x, m.disj3))
    agg_cons = hull.get_disaggregation_constraint(m.disj1.y, m.disjunction2)
    assertExpressionsEqual(self, agg_cons.expr, m.disj1.y == y1 + hull.get_disaggregated_var(m.disj1.y, m.disj4))
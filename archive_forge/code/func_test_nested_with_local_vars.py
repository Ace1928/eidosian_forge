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
def test_nested_with_local_vars(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))
    m.S = RangeSet(2)

    @m.Disjunct()
    def d_l(d):
        d.lambdas = Var(m.S, bounds=(0, 1))
        d.LocalVars = Suffix(direction=Suffix.LOCAL)
        d.LocalVars[d] = list(d.lambdas.values())
        d.c1 = Constraint(expr=d.lambdas[1] + d.lambdas[2] == 1)
        d.c2 = Constraint(expr=m.x == 2 * d.lambdas[1] + 3 * d.lambdas[2])

    @m.Disjunct()
    def d_r(d):

        @d.Disjunct()
        def d_l(e):
            e.lambdas = Var(m.S, bounds=(0, 1))
            e.LocalVars = Suffix(direction=Suffix.LOCAL)
            e.LocalVars[e] = list(e.lambdas.values())
            e.c1 = Constraint(expr=e.lambdas[1] + e.lambdas[2] == 1)
            e.c2 = Constraint(expr=m.x == 2 * e.lambdas[1] + 3 * e.lambdas[2])

        @d.Disjunct()
        def d_r(e):
            e.lambdas = Var(m.S, bounds=(0, 1))
            e.LocalVars = Suffix(direction=Suffix.LOCAL)
            e.LocalVars[e] = list(e.lambdas.values())
            e.c1 = Constraint(expr=e.lambdas[1] + e.lambdas[2] == 1)
            e.c2 = Constraint(expr=m.x == 2 * e.lambdas[1] + 3 * e.lambdas[2])
        d.LocalVars = Suffix(direction=Suffix.LOCAL)
        d.LocalVars[d] = [d.d_l.indicator_var.get_associated_binary(), d.d_r.indicator_var.get_associated_binary()]
        d.inner_disj = Disjunction(expr=[d.d_l, d.d_r])
    m.disj = Disjunction(expr=[m.d_l, m.d_r])
    m.obj = Objective(expr=m.x)
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    x1 = hull.get_disaggregated_var(m.x, m.d_l)
    x2 = hull.get_disaggregated_var(m.x, m.d_r)
    x3 = hull.get_disaggregated_var(m.x, m.d_r.d_l)
    x4 = hull.get_disaggregated_var(m.x, m.d_r.d_r)
    for d, x in [(m.d_l, x1), (m.d_r.d_l, x3), (m.d_r.d_r, x4)]:
        lambda1 = hull.get_disaggregated_var(d.lambdas[1], d)
        self.assertIs(lambda1, d.lambdas[1])
        lambda2 = hull.get_disaggregated_var(d.lambdas[2], d)
        self.assertIs(lambda2, d.lambdas[2])
        cons = hull.get_transformed_constraints(d.c1)
        self.assertEqual(len(cons), 1)
        convex_combo = cons[0]
        convex_combo_expr = self.simplify_cons(convex_combo)
        assertExpressionsEqual(self, convex_combo_expr, lambda1 + lambda2 - d.indicator_var.get_associated_binary() == 0.0)
        cons = hull.get_transformed_constraints(d.c2)
        self.assertEqual(len(cons), 1)
        get_x = cons[0]
        get_x_expr = self.simplify_cons(get_x)
        assertExpressionsEqual(self, get_x_expr, x - 2 * lambda1 - 3 * lambda2 == 0.0)
    cons = hull.get_disaggregation_constraint(m.x, m.disj)
    assertExpressionsEqual(self, cons.expr, m.x == x1 + x2)
    cons = hull.get_disaggregation_constraint(m.x, m.d_r.inner_disj)
    cons_expr = self.simplify_cons(cons)
    assertExpressionsEqual(self, cons_expr, x2 - x3 - x4 == 0.0)
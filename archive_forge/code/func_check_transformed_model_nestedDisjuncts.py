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
def check_transformed_model_nestedDisjuncts(self, m, d3, d4):
    hull = TransformationFactory('gdp.hull')
    transBlock = m._pyomo_gdp_hull_reformulation
    self.assertTrue(transBlock.active)
    xor = transBlock.disj_xor
    self.assertIsInstance(xor, Constraint)
    ct.check_obj_in_active_tree(self, xor)
    assertExpressionsEqual(self, xor.expr, m.d1.binary_indicator_var + m.d2.binary_indicator_var == 1)
    self.assertIs(xor, m.disj.algebraic_constraint)
    self.assertIs(m.disj, hull.get_src_disjunction(xor))
    xor = m.d1.disj2.algebraic_constraint
    self.assertIs(m.d1.disj2, hull.get_src_disjunction(xor))
    xor = hull.get_transformed_constraints(xor)
    self.assertEqual(len(xor), 1)
    xor = xor[0]
    ct.check_obj_in_active_tree(self, xor)
    xor_expr = self.simplify_cons(xor)
    assertExpressionsEqual(self, xor_expr, d3 + d4 - m.d1.binary_indicator_var == 0.0)
    x_d3 = hull.get_disaggregated_var(m.x, m.d1.d3)
    x_d4 = hull.get_disaggregated_var(m.x, m.d1.d4)
    x_d1 = hull.get_disaggregated_var(m.x, m.d1)
    x_d2 = hull.get_disaggregated_var(m.x, m.d2)
    for x in [x_d1, x_d2, x_d3, x_d4]:
        self.assertEqual(x.lb, 0)
        self.assertEqual(x.ub, 2)
    cons = hull.get_disaggregation_constraint(m.x, m.d1.disj2)
    ct.check_obj_in_active_tree(self, cons)
    cons_expr = self.simplify_cons(cons)
    assertExpressionsEqual(self, cons_expr, x_d1 - x_d3 - x_d4 == 0.0)
    cons = hull.get_disaggregation_constraint(m.x, m.disj)
    ct.check_obj_in_active_tree(self, cons)
    cons_expr = self.simplify_cons(cons)
    assertExpressionsEqual(self, cons_expr, m.x - x_d1 - x_d2 == 0.0)
    cons = hull.get_transformed_constraints(m.d1.d3.c)
    self.assertEqual(len(cons), 1)
    cons = cons[0]
    ct.check_obj_in_active_tree(self, cons)
    cons_expr = self.simplify_leq_cons(cons)
    assertExpressionsEqual(self, cons_expr, 1.2 * d3 - x_d3 <= 0.0)
    cons = hull.get_transformed_constraints(m.d1.d4.c)
    self.assertEqual(len(cons), 1)
    cons = cons[0]
    ct.check_obj_in_active_tree(self, cons)
    cons_expr = self.simplify_leq_cons(cons)
    assertExpressionsEqual(self, cons_expr, 1.3 * d4 - x_d4 <= 0.0)
    cons = hull.get_transformed_constraints(m.d1.c)
    self.assertEqual(len(cons), 1)
    cons = cons[0]
    ct.check_obj_in_active_tree(self, cons)
    cons_expr = self.simplify_leq_cons(cons)
    assertExpressionsEqual(self, cons_expr, 1.0 * m.d1.binary_indicator_var - x_d1 <= 0.0)
    cons = hull.get_transformed_constraints(m.d2.c)
    self.assertEqual(len(cons), 1)
    cons = cons[0]
    ct.check_obj_in_active_tree(self, cons)
    cons_expr = self.simplify_leq_cons(cons)
    assertExpressionsEqual(self, cons_expr, 1.1 * m.d2.binary_indicator_var - x_d2 <= 0.0)
    cons = hull.get_var_bounds_constraint(x_d1)
    self.assertEqual(len(cons), 1)
    ct.check_obj_in_active_tree(self, cons['ub'])
    cons_expr = self.simplify_leq_cons(cons['ub'])
    assertExpressionsEqual(self, cons_expr, x_d1 - 2 * m.d1.binary_indicator_var <= 0.0)
    cons = hull.get_var_bounds_constraint(x_d2)
    self.assertEqual(len(cons), 1)
    ct.check_obj_in_active_tree(self, cons['ub'])
    cons_expr = self.simplify_leq_cons(cons['ub'])
    assertExpressionsEqual(self, cons_expr, x_d2 - 2 * m.d2.binary_indicator_var <= 0.0)
    cons = hull.get_var_bounds_constraint(x_d3, m.d1.d3)
    self.assertEqual(len(cons), 1)
    cons = hull.get_transformed_constraints(cons['ub'])
    self.assertEqual(len(cons), 1)
    ub = cons[0]
    ct.check_obj_in_active_tree(self, ub)
    cons_expr = self.simplify_leq_cons(ub)
    assertExpressionsEqual(self, cons_expr, x_d3 - 2 * d3 <= 0.0)
    cons = hull.get_var_bounds_constraint(x_d4, m.d1.d4)
    self.assertEqual(len(cons), 1)
    cons = hull.get_transformed_constraints(cons['ub'])
    self.assertEqual(len(cons), 1)
    ub = cons[0]
    ct.check_obj_in_active_tree(self, ub)
    cons_expr = self.simplify_leq_cons(ub)
    assertExpressionsEqual(self, cons_expr, x_d4 - 2 * d4 <= 0.0)
    cons = hull.get_var_bounds_constraint(x_d3, m.d1)
    self.assertEqual(len(cons), 1)
    ub = cons['ub']
    ct.check_obj_in_active_tree(self, ub)
    cons_expr = self.simplify_leq_cons(ub)
    assertExpressionsEqual(self, cons_expr, x_d3 - 2 * m.d1.binary_indicator_var <= 0.0)
    cons = hull.get_var_bounds_constraint(x_d4, m.d1)
    self.assertEqual(len(cons), 1)
    ub = cons['ub']
    ct.check_obj_in_active_tree(self, ub)
    cons_expr = self.simplify_leq_cons(ub)
    assertExpressionsEqual(self, cons_expr, x_d4 - 2 * m.d1.binary_indicator_var <= 0.0)
    cons = hull.get_var_bounds_constraint(d3)
    ct.check_obj_in_active_tree(self, cons['ub'])
    assertExpressionsEqual(self, cons['ub'].expr, d3 <= m.d1.binary_indicator_var)
    cons = hull.get_var_bounds_constraint(d4)
    ct.check_obj_in_active_tree(self, cons['ub'])
    assertExpressionsEqual(self, cons['ub'].expr, d4 <= m.d1.binary_indicator_var)
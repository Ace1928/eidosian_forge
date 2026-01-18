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
def check_transformation_blocks_nestedDisjunctions(self, m, transformation):
    disjunctionTransBlock = m.disj.algebraic_constraint.parent_block()
    transBlocks = disjunctionTransBlock.relaxedDisjuncts
    if transformation == 'bigm':
        self.assertEqual(len(transBlocks), 4)
        self.assertIs(transBlocks[0], m.d1.d3.transformation_block)
        self.assertIs(transBlocks[1], m.d1.d4.transformation_block)
        self.assertIs(transBlocks[2], m.d1.transformation_block)
        self.assertIs(transBlocks[3], m.d2.transformation_block)
    if transformation == 'hull':
        hull = TransformationFactory('gdp.hull')
        d3 = hull.get_disaggregated_var(m.d1.d3.binary_indicator_var, m.d1)
        d4 = hull.get_disaggregated_var(m.d1.d4.binary_indicator_var, m.d1)
        self.check_transformed_model_nestedDisjuncts(m, d3, d4)
        d32 = hull.get_disaggregated_var(m.d1.d3.binary_indicator_var, m.d2)
        d42 = hull.get_disaggregated_var(m.d1.d4.binary_indicator_var, m.d2)
        cons = hull.get_var_bounds_constraint(d32)
        self.assertEqual(len(cons), 1)
        check_obj_in_active_tree(self, cons['ub'])
        cons_expr = self.simplify_leq_cons(cons['ub'])
        assertExpressionsEqual(self, cons_expr, d32 + m.d1.binary_indicator_var - 1 <= 0.0)
        cons = hull.get_var_bounds_constraint(d42)
        self.assertEqual(len(cons), 1)
        check_obj_in_active_tree(self, cons['ub'])
        cons_expr = self.simplify_leq_cons(cons['ub'])
        assertExpressionsEqual(self, cons_expr, d42 + m.d1.binary_indicator_var - 1 <= 0.0)
        cons = hull.get_disaggregation_constraint(m.d1.d3.binary_indicator_var, m.disj)
        check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, m.d1.d3.binary_indicator_var - d32 - d3 == 0.0)
        cons = hull.get_disaggregation_constraint(m.d1.d4.binary_indicator_var, m.disj)
        check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, m.d1.d4.binary_indicator_var - d42 - d4 == 0.0)
        num_cons = len(list(m.component_data_objects(Constraint, active=True, descend_into=Block)))
        self.assertEqual(num_cons, 20)
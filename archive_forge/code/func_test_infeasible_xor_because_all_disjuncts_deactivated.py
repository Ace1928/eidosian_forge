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
def test_infeasible_xor_because_all_disjuncts_deactivated(self):
    m = ct.setup_infeasible_xor_because_all_disjuncts_deactivated(self, 'hull')
    hull = TransformationFactory('gdp.hull')
    transBlock = m.component('_pyomo_gdp_hull_reformulation')
    self.assertIsInstance(transBlock, Block)
    self.assertEqual(len(transBlock.relaxedDisjuncts), 2)
    self.assertIsInstance(transBlock.component('disjunction_xor'), Constraint)
    disjunct1 = transBlock.relaxedDisjuncts[0]
    d3_ind = m.disjunction_disjuncts[0].nestedDisjunction_disjuncts[0].binary_indicator_var
    d4_ind = m.disjunction_disjuncts[0].nestedDisjunction_disjuncts[1].binary_indicator_var
    d3_ind_dis = disjunct1.disaggregatedVars.component('disjunction_disjuncts[0].nestedDisjunction_disjuncts[0].binary_indicator_var')
    self.assertIs(hull.get_disaggregated_var(d3_ind, m.disjunction_disjuncts[0]), d3_ind_dis)
    self.assertIs(hull.get_src_var(d3_ind_dis), d3_ind)
    d4_ind_dis = disjunct1.disaggregatedVars.component('disjunction_disjuncts[0].nestedDisjunction_disjuncts[1].binary_indicator_var')
    self.assertIs(hull.get_disaggregated_var(d4_ind, m.disjunction_disjuncts[0]), d4_ind_dis)
    self.assertIs(hull.get_src_var(d4_ind_dis), d4_ind)
    relaxed_xor = hull.get_transformed_constraints(m.disjunction_disjuncts[0].nestedDisjunction.algebraic_constraint)
    self.assertEqual(len(relaxed_xor), 1)
    relaxed_xor = relaxed_xor[0]
    repn = generate_standard_repn(relaxed_xor.body)
    self.assertEqual(value(relaxed_xor.lower), 0)
    self.assertEqual(value(relaxed_xor.upper), 0)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 3)
    ct.check_linear_coef(self, repn, m.disjunction.disjuncts[0].indicator_var, -1)
    ct.check_linear_coef(self, repn, d3_ind_dis, 1)
    ct.check_linear_coef(self, repn, d4_ind_dis, 1)
    self.assertEqual(repn.constant, 0)
    d3_ind_dis_cons = transBlock.disaggregationConstraints[1]
    self.assertEqual(d3_ind_dis_cons.lower, 0)
    self.assertEqual(d3_ind_dis_cons.upper, 0)
    repn = generate_standard_repn(d3_ind_dis_cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.constant, 0)
    ct.check_linear_coef(self, repn, d3_ind_dis, -1)
    ct.check_linear_coef(self, repn, transBlock._disaggregatedVars[0], -1)
    d4_ind_dis_cons = transBlock.disaggregationConstraints[2]
    self.assertEqual(d4_ind_dis_cons.lower, 0)
    self.assertEqual(d4_ind_dis_cons.upper, 0)
    repn = generate_standard_repn(d4_ind_dis_cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.constant, 0)
    ct.check_linear_coef(self, repn, d4_ind_dis, -1)
    ct.check_linear_coef(self, repn, transBlock._disaggregatedVars[1], -1)
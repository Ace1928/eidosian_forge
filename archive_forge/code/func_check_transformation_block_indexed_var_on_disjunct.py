import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.logical_expr import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers
def check_transformation_block_indexed_var_on_disjunct(self, m, original_disjunction):
    b, disj1, disj2 = self.check_transformation_block_disjuncts_and_constraints(m, original_disjunction)
    self.assertEqual(len(disj1.component_map(Var)), 4)
    self.assertEqual(len(disj2.component_map(Var)), 3)
    aux_vars1 = disj1.component('disj1.c_aux_vars')
    aux_vars2 = disj2.component('disj2.c_aux_vars')
    self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -72, 96, -72, 96)
    c = disj1.component('disj1.c')
    self.assertEqual(len(c), 1)
    c1 = c[0]
    self.check_disj_constraint(c1, 1, aux_vars1[0], aux_vars1[1])
    c = disj2.component('disj2.c')
    self.assertEqual(len(c), 1)
    c2 = c[0]
    self.check_disj_constraint(c2, -35, aux_vars2[0], aux_vars2[1])
    c = b.component('disj1.c_split_constraints')
    self.assertEqual(len(c), 2)
    c1 = c[0]
    self.check_global_constraint_disj1(c1, aux_vars1[0], m.disj1.x[1], m.disj1.x[2])
    c2 = c[1]
    self.check_global_constraint_disj1(c2, aux_vars1[1], m.disj1.x[3], m.disj1.x[4])
    c = b.component('disj2.c_split_constraints')
    self.assertEqual(len(c), 2)
    c1 = c[0]
    self.check_global_constraint_disj2(c1, aux_vars2[0], m.disj1.x[1], m.disj1.x[2])
    c2 = c[1]
    self.check_global_constraint_disj2(c2, aux_vars2[1], m.disj1.x[3], m.disj1.x[4])
    return (b, disj1, disj2)
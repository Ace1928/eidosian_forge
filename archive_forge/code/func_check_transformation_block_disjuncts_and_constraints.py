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
def check_transformation_block_disjuncts_and_constraints(self, m, original_disjunction, disjunction_name=None):
    b = m.component('_pyomo_gdp_partition_disjuncts_reformulation')
    self.assertIsInstance(b, Block)
    self.assertEqual(len(b.component_map(Disjunction)), 1)
    self.assertEqual(len(b.component_map(Disjunct)), 2)
    self.assertEqual(len(b.component_map(Constraint)), 2)
    self.assertEqual(len(b.component_map(LogicalConstraint)), 1)
    if disjunction_name is None:
        disjunction = b.disjunction
    else:
        disjunction = b.component(disjunction_name)
    self.assertEqual(len(disjunction.disjuncts), 2)
    disj1 = disjunction.disjuncts[0]
    disj2 = disjunction.disjuncts[1]
    self.assertEqual(len(disj1.component_map(Constraint)), 1)
    self.assertEqual(len(disj2.component_map(Constraint)), 1)
    equivalence = b.component('indicator_var_equalities')
    self.assertIsInstance(equivalence, LogicalConstraint)
    self.assertEqual(len(equivalence), 2)
    for i, variables in enumerate([(original_disjunction.disjuncts[0].indicator_var, disj1.indicator_var), (original_disjunction.disjuncts[1].indicator_var, disj2.indicator_var)]):
        cons = equivalence[i]
        self.assertIsInstance(cons.body, EquivalenceExpression)
        self.assertIs(cons.body.args[0], variables[0])
        self.assertIs(cons.body.args[1], variables[1])
    return (b, disj1, disj2)
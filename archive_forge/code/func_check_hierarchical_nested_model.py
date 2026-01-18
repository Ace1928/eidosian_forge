from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
def check_hierarchical_nested_model(self, m, bigm):
    outer_xor = m.disjunction_block.disjunction.algebraic_constraint
    ct.check_two_term_disjunction_xor(self, outer_xor, m.disj1, m.disjunct_block.disj2)
    self.check_inner_xor_constraint(m.disjunct_block.disj2.disjunction, m.disjunct_block.disj2, bigm)
    disj1c = bigm.get_transformed_constraints(m.disj1.c)
    self.check_first_disjunct_constraint(disj1c, m.x, m.disj1.binary_indicator_var)
    disj2c = bigm.get_transformed_constraints(m.disjunct_block.disj2.c)
    self.check_second_disjunct_constraint(disj2c, m.x, m.disjunct_block.disj2.binary_indicator_var)
    innerd1c = bigm.get_transformed_constraints(m.disjunct_block.disj2.disjunction_disjuncts[0].constraint[1])
    self.check_first_disjunct_constraint(innerd1c, m.x, m.disjunct_block.disj2.disjunction_disjuncts[0].binary_indicator_var)
    innerd2c = bigm.get_transformed_constraints(m.disjunct_block.disj2.disjunction_disjuncts[1].constraint[1])
    self.check_second_disjunct_constraint(innerd2c, m.x, m.disjunct_block.disj2.disjunction_disjuncts[1].binary_indicator_var)
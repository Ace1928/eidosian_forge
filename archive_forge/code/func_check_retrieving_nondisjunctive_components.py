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
def check_retrieving_nondisjunctive_components(self, transformation):
    m = models.makeTwoTermDisj()
    m.b = Block()
    m.b.global_cons = Constraint(expr=m.a + m.x >= 8)
    m.another_global_cons = Constraint(expr=m.a + m.x <= 11)
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)
    self.assertRaisesRegex(GDP_Error, "Constraint 'b.global_cons' is not on a disjunct and so was not transformed", trans.get_transformed_constraints, m.b.global_cons)
    self.assertRaisesRegex(GDP_Error, "Constraint 'b.global_cons' is not a transformed constraint", trans.get_src_constraint, m.b.global_cons)
    self.assertRaisesRegex(GDP_Error, "Constraint 'another_global_cons' is not a transformed constraint", trans.get_src_constraint, m.another_global_cons)
    self.assertRaisesRegex(GDP_Error, "Block 'b' doesn't appear to be a transformation block for a disjunct. No source disjunct found.", trans.get_src_disjunct, m.b)
    self.assertRaisesRegex(GDP_Error, "It appears that 'another_global_cons' is not an XOR or OR constraint resulting from transforming a Disjunction.", trans.get_src_disjunction, m.another_global_cons)
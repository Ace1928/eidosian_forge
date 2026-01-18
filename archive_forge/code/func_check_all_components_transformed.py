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
def check_all_components_transformed(self, m):
    self.assertIsInstance(m.disj.algebraic_constraint, Constraint)
    self.assertIsInstance(m.d1.disj2.algebraic_constraint, Constraint)
    self.assertIsInstance(m.d1.transformation_block, _BlockData)
    self.assertIsInstance(m.d2.transformation_block, _BlockData)
    self.assertIsInstance(m.d1.d3.transformation_block, _BlockData)
    self.assertIsInstance(m.d1.d4.transformation_block, _BlockData)
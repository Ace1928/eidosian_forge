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
def check_obj_in_active_tree(self, obj, root=None):
    self.assertTrue(obj.active)
    parent = obj.parent_component()
    self.assertTrue(parent.active)
    blk = parent.parent_block()
    while blk is not root:
        self.assertTrue(blk.active)
        blk = blk.parent_block()
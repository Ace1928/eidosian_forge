import os
from filecmp import cmp
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.collections import OrderedSet
from pyomo.common.fileutils import this_file_dir
import pyomo.core.expr as EXPR
from pyomo.core.base import SymbolMap
from pyomo.environ import (
from pyomo.repn.plugins.baron_writer import expression_to_string
def test_invalid_suffix(self):
    m = ConcreteModel()
    m.x = Var(within=Binary)
    m.y = Var([1, 2, 3], within=Binary)
    m.c = Constraint(expr=m.y[1] * m.y[2] - 2 * m.x >= 0)
    m.obj = Objective(expr=m.y[1] + m.y[2], sense=maximize)
    m.priorities = Suffix(direction=Suffix.EXPORT)
    m.priorities[m.x] = 1
    m.priorities[m.y] = 2
    with self.assertRaisesRegex(ValueError, "The BARON writer can not export suffix with name 'priorities'. Either remove it from the model or deactivate it."):
        m.write(StringIO(), format='bar')
    m._name = 'TestModel'
    with self.assertRaisesRegex(ValueError, "The BARON writer can not export suffix with name 'priorities'. Either remove it from the model 'TestModel' or deactivate it."):
        m.write(StringIO(), format='bar')
    p = m.priorities
    del m.priorities
    m.blk = Block()
    m.blk.sub = Block()
    m.blk.sub.priorities = p
    with self.assertRaisesRegex(ValueError, "The BARON writer can not export suffix with name 'priorities'. Either remove it from the block 'blk.sub' or deactivate it."):
        m.write(StringIO(), format='bar')
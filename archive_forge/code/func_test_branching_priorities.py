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
def test_branching_priorities(self):
    m = ConcreteModel()
    m.x = Var(within=Binary)
    m.y = Var([1, 2, 3], within=Binary)
    m.c = Constraint(expr=m.y[1] * m.y[2] - 2 * m.x >= 0)
    m.obj = Objective(expr=m.y[1] + m.y[2], sense=maximize)
    m.priority = Suffix(direction=Suffix.EXPORT)
    m.priority[m.x] = 1
    m.priority[m.y] = 2
    self._check_baseline(m)
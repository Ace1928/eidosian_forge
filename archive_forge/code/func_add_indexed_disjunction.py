import pyomo.common.unittest as unittest
from pyomo.core import ConcreteModel, Var, Expression, Block, RangeSet, Any
import pyomo.core.expr as EXPR
from pyomo.core.base.expression import _ExpressionData
from pyomo.gdp.util import (
from pyomo.gdp import Disjunct, Disjunction
def add_indexed_disjunction(self, parent, m):
    parent.indexed = Disjunction(Any)
    parent.indexed[1] = [[sum((m.x[i] ** 2 for i in m.I)) <= 1], [sum(((3 - m.x[i]) ** 2 for i in m.I)) <= 1]]
    parent.indexed[0] = [[(m.x[1] - 1) ** 2 + m.x[2] ** 2 <= 1], [-(m.x[1] - 2) ** 2 - (m.x[2] - 3) ** 2 >= -1]]
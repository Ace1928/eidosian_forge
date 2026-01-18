import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_cnf(self):
    m = ConcreteModel()
    m.Y1 = BooleanVar()
    m.Y2 = BooleanVar()
    implication = implies(m.Y1, m.Y2)
    x = to_cnf(implication)[0]
    _check_equivalent(self, implication, x)
    atleast_expr = atleast(1, m.Y1, m.Y2)
    x = to_cnf(atleast_expr)[0]
    self.assertIs(atleast_expr, x)
    nestedatleast = implies(m.Y1, atleast_expr)
    m.extraY = BooleanVarList()
    indicator_map = ComponentMap()
    x = to_cnf(nestedatleast, m.extraY, indicator_map)
    self.assertEqual(str(x[0]), 'extraY[1] ∨ ~Y1')
    self.assertIs(indicator_map[m.extraY[1]], atleast_expr)
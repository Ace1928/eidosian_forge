import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_indirection_multi_index_neither_constant_diff_vars(self):
    m = self.get_model()
    m.z = Var(m.I, m.I, domain=Integers)
    m.y = Var(within=[1, 3, 5])
    e = m.z[m.x, m.y]
    visitor = self.get_visitor()
    expr = visitor.walk_expression((e, e, 0))
    z = {}
    for i in [6, 7, 8]:
        for j in [1, 3, 5]:
            self.assertIn(id(m.z[i, 3]), visitor.var_map)
            z[i, j] = visitor.var_map[id(m.z[i, j])]
    self.assertIn(id(m.x), visitor.var_map)
    x = visitor.var_map[id(m.x)]
    self.assertIn(id(m.y), visitor.var_map)
    y = visitor.var_map[id(m.y)]
    self.assertTrue(expr[1].equals(cp.element([z[i, j] for i in [6, 7, 8] for j in [1, 3, 5]], 0 + 1 * (x - 6) // 1 + 3 * (y - 1) // 2)))
import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_indirection_multi_index_second_constant(self):
    m = self.get_model()
    m.z = Var(m.I, m.I, domain=Integers)
    e = m.z[m.x, 3]
    visitor = self.get_visitor()
    expr = visitor.walk_expression((e, e, 0))
    z = {}
    for i in [6, 7, 8]:
        self.assertIn(id(m.z[i, 3]), visitor.var_map)
        z[i, 3] = visitor.var_map[id(m.z[i, 3])]
    self.assertIn(id(m.x), visitor.var_map)
    x = visitor.var_map[id(m.x)]
    self.assertTrue(expr[1].equals(cp.element([z[i, 3] for i in [6, 7, 8]], 0 + 1 * (x - 6) // 1)))
import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_indirection_with_param(self):
    m = ConcreteModel()

    def param_rule(m, i):
        return i + 1
    m.p = Param([1, 3, 5], initialize=param_rule)
    m.x = Var(within={1, 3, 5})
    m.a = Var(domain=Integers, bounds=(0, 100))
    e = m.p[m.x] / m.a
    visitor = self.get_visitor()
    expr = visitor.walk_expression((e, e, 0))
    self.assertIn(id(m.x), visitor.var_map)
    self.assertIn(id(m.a), visitor.var_map)
    x = visitor.var_map[id(m.x)]
    a = visitor.var_map[id(m.a)]
    self.assertTrue(expr[1].equals(cp.element([2, 4, 6], 0 + 1 * (x - 1) // 2) / a))
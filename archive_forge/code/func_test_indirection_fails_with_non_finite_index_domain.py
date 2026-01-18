import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_indirection_fails_with_non_finite_index_domain(self):
    m = self.get_model()
    m.a.domain = Integers
    m.x.setlb(None)
    m.x.setub(None)
    m.c = Constraint(expr=m.a[m.x] >= 0)
    visitor = self.get_visitor()
    with self.assertRaisesRegex(ValueError, "Variable indirection 'a\\[x\\]' contains argument 'x', which is not restricted to a finite discrete domain"):
        expr = visitor.walk_expression((m.c.body, m.c, 0))
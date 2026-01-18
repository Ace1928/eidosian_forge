import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_indirection_invalid_index_domain(self):
    m = self.get_model()
    m.a.domain = Integers
    m.a.bounds = (6, 8)
    m.y = Var(within=Integers, bounds=(0, 10))
    e = m.a[m.y]
    visitor = self.get_visitor()
    with self.assertRaisesRegex(ValueError, "Variable indirection 'a\\[y\\]' permits an index '0' that is not a valid key."):
        expr = visitor.walk_expression((e, e, 0))
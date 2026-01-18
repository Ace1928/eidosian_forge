import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_infinite_domain_var(self):
    m = ConcreteModel()
    m.Evens = RangeSet(ranges=(NumericRange(0, None, 2), NumericRange(0, None, -2)))
    m.x = Var(domain=m.Evens)
    e = m.x ** 2
    visitor = self.get_visitor()
    with self.assertRaisesRegex(ValueError, "The LogicalToDoCplex writer does not support infinite discrete domains. Cannot write Var 'x' with domain 'Evens'"):
        expr = visitor.walk_expression((e, e, 0))
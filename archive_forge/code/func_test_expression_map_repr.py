import sympy
import cirq
from cirq.study import flatten_expressions
def test_expression_map_repr():
    cirq.testing.assert_equivalent_repr(cirq.ExpressionMap({'a': 'b'}))
import sympy
import cirq
from cirq.study import flatten_expressions
def test_flattener_repr():
    assert repr(flatten_expressions._ParamFlattener({'a': 1})) == '_ParamFlattener({a: 1})'
    assert repr(flatten_expressions._ParamFlattener({'a': 1}, get_param_name=lambda expr: 'x')).startswith('_ParamFlattener({a: 1}, get_param_name=<function ')
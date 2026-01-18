import sympy
import cirq
from cirq.study import flatten_expressions
def test_flattener_value_of():
    flattener = flatten_expressions._ParamFlattener({'c': 5, 'x1': 'x1'})
    assert flattener.value_of(9) == 9
    assert flattener.value_of('c') == 5
    assert flattener.value_of(sympy.Symbol('c')) == 5
    assert flattener.value_of(sympy.Symbol('c') / 2 + 1) == sympy.Symbol('<c/2 + 1>')
    assert flattener.value_of(sympy.Symbol('c') / 2 + 1) == sympy.Symbol('<c/2 + 1>')
    assert flattener.value_of(sympy.Symbol('c') / sympy.Symbol('2 + 1')) == sympy.Symbol('<c/2 + 1>_1')
    assert flattener.value_of(sympy.Symbol('c/2') + 1) == sympy.Symbol('<c/2 + 1>_2')
    assert cirq.flatten([sympy.Symbol('c') / 2 + 1, sympy.Symbol('c/2') + 1])[0] == [sympy.Symbol('<c/2 + 1>'), sympy.Symbol('<c/2 + 1>_1')]
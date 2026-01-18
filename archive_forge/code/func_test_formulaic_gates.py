import sympy
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_formulaic_gates():
    a, b = cirq.LineQubit.range(2)
    t = sympy.Symbol('t')
    assert_url_to_circuit_returns('{"cols":[["X^ft",{"id":"X^ft","arg":"t*t"}]]}', cirq.Circuit(cirq.X(a) ** sympy.sin(sympy.pi * t), cirq.X(b) ** (t * t)))
    assert_url_to_circuit_returns('{"cols":[["Y^ft",{"id":"Y^ft","arg":"t*t"}]]}', cirq.Circuit(cirq.Y(a) ** sympy.sin(sympy.pi * t), cirq.Y(b) ** (t * t)))
    assert_url_to_circuit_returns('{"cols":[["Z^ft",{"id":"Z^ft","arg":"t*t"}]]}', cirq.Circuit(cirq.Z(a) ** sympy.sin(sympy.pi * t), cirq.Z(b) ** (t * t)))
    assert_url_to_circuit_returns('{"cols":[["Rxft",{"id":"Rxft","arg":"t*t"}]]}', cirq.Circuit(cirq.rx(sympy.pi * t * t).on(a), cirq.rx(t * t).on(b)))
    assert_url_to_circuit_returns('{"cols":[["Ryft",{"id":"Ryft","arg":"t*t"}]]}', cirq.Circuit(cirq.ry(sympy.pi * t * t).on(a), cirq.ry(t * t).on(b)))
    assert_url_to_circuit_returns('{"cols":[["Rzft",{"id":"Rzft","arg":"t*t"}]]}', cirq.Circuit(cirq.rz(sympy.pi * t * t).on(a), cirq.rz(t * t).on(b)))
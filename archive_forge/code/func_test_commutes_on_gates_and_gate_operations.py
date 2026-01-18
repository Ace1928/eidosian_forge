import numpy as np
import pytest
import sympy
import cirq
def test_commutes_on_gates_and_gate_operations():
    X, Y, Z = tuple((cirq.unitary(A) for A in (cirq.X, cirq.Y, cirq.Z)))
    XGate, YGate, ZGate = (cirq.MatrixGate(A) for A in (X, Y, Z))
    XXGate, YYGate, ZZGate = (cirq.MatrixGate(cirq.kron(A, A)) for A in (X, Y, Z))
    a, b = cirq.LineQubit.range(2)
    for A in (XGate, YGate, ZGate):
        assert cirq.commutes(A, A)
        assert A._commutes_on_qids_(a, A, atol=1e-08) is NotImplemented
        with pytest.raises(TypeError):
            cirq.commutes(A(a), A)
        with pytest.raises(TypeError):
            cirq.commutes(A, A(a))
        assert cirq.commutes(A(a), A(a))
        assert cirq.commutes(A, XXGate, default='default') == 'default'
    for A, B in [(XGate, YGate), (XGate, ZGate), (ZGate, YGate), (XGate, cirq.Y), (XGate, cirq.Z), (ZGate, cirq.Y)]:
        assert not cirq.commutes(A, B)
        assert cirq.commutes(A(a), B(b))
        assert not cirq.commutes(A(a), B(a))
        with pytest.raises(TypeError):
            cirq.commutes(A, B(a))
        cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(A, B)
    for A, B in [(XXGate, YYGate), (XXGate, ZZGate)]:
        assert cirq.commutes(A, B)
        with pytest.raises(TypeError):
            cirq.commutes(A(a, b), B)
        with pytest.raises(TypeError):
            cirq.commutes(A, B(a, b))
        assert cirq.commutes(A(a, b), B(a, b))
        assert cirq.definitely_commutes(A(a, b), B(a, b))
        cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(A, B)
    for A, B in [(XGate, XXGate), (XGate, YYGate)]:
        with pytest.raises(TypeError):
            cirq.commutes(A, B(a, b))
        assert not cirq.definitely_commutes(A, B(a, b))
        with pytest.raises(TypeError):
            assert cirq.commutes(A(b), B)
        with pytest.raises(TypeError):
            assert cirq.commutes(A, B)
        cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(A, B)
    with pytest.raises(TypeError):
        assert cirq.commutes(XGate, cirq.X ** sympy.Symbol('e'))
    with pytest.raises(TypeError):
        assert cirq.commutes(XGate(a), 'Gate')
    assert cirq.commutes(XGate(a), 'Gate', default='default') == 'default'
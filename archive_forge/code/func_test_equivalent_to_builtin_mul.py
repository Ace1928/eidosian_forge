import pytest
import sympy
import cirq
def test_equivalent_to_builtin_mul():
    test_vals = [0, 1, 1j, -2.5, Neither(), MulReturnsNotImplemented(), RMulReturnsNotImplemented(), MulReturnsFive(), RMulReturnsSix(), MulSevenRMulEight()]
    for a in test_vals:
        for b in test_vals:
            if type(a) == type(b) == RMulReturnsSix:
                continue
            c = cirq.mul(a, b, default=None)
            if c is None:
                with pytest.raises(TypeError):
                    _ = a * b
                with pytest.raises(TypeError):
                    _ = cirq.mul(a, b)
            else:
                assert c == a * b
import itertools
import numpy as np
import pytest
import cirq
def test_identity_multiplication():
    a, b, c = cirq.LineQubit.range(3)
    assert cirq.X(a) * cirq.I(a) == cirq.X(a)
    assert cirq.X(a) * cirq.I(b) == cirq.X(a)
    assert cirq.X(a) * cirq.Y(b) * cirq.I(c) == cirq.X(a) * cirq.Y(b)
    assert cirq.I(c) * cirq.X(a) * cirq.Y(b) == cirq.X(a) * cirq.Y(b)
    with pytest.raises(TypeError):
        _ = cirq.H(c) * cirq.X(a) * cirq.Y(b)
    with pytest.raises(TypeError):
        _ = cirq.X(a) * cirq.Y(b) * cirq.H(c)
    with pytest.raises(TypeError):
        _ = cirq.I(a) * str(cirq.Y(b))
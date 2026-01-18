import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('gate', [cirq.X ** sympy.Symbol('a'), cirq.testing.SingleQubitGate()])
def test_no_unitary(gate):
    g = cirq.ParallelGate(gate, 2)
    assert not cirq.has_unitary(g)
    assert cirq.unitary(g, None) is None
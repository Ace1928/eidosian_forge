import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_non_clifford_circuit():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    circuit.append(cirq.T(q0))
    with pytest.raises(TypeError, match='support cirq.T'):
        cirq.CliffordSimulator().simulate(circuit)
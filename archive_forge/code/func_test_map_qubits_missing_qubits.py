import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_map_qubits_missing_qubits():
    q0, q1, q2 = _make_qubits(3)
    qubit_map = {q1: q2}
    before = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Y}), exponent_neg=0.1)
    with pytest.raises(ValueError, match='have a key'):
        _ = before.map_qubits(qubit_map)
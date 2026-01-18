import numpy as np
import pytest
import cirq
def test_basis_construction():
    states = []
    for gate in [cirq.X, cirq.Y, cirq.Z]:
        for e_val in [+1, -1]:
            states.append(gate.basis[e_val])
    assert states == cirq.PAULI_STATES
import pytest
import numpy as np
import cirq
from cirq.testing import assert_allclose_up_to_global_phase
def test_misaligned_qubits():
    qubits = cirq.LineQubit.range(1)
    tableau = cirq.CliffordTableau(num_qubits=2)
    with pytest.raises(ValueError):
        cirq.decompose_clifford_tableau_to_operations(qubits, tableau)
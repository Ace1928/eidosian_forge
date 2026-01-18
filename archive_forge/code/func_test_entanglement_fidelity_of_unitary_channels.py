import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('gate, expected_entanglement_fidelity', ((cirq.I, 1), (cirq.X, 0), (cirq.Y, 0), (cirq.Z, 0), (cirq.S, 1 / 2), (cirq.CNOT, 1 / 4), (cirq.TOFFOLI, 9 / 16)))
def test_entanglement_fidelity_of_unitary_channels(gate, expected_entanglement_fidelity):
    assert np.isclose(cirq.entanglement_fidelity(gate), expected_entanglement_fidelity)
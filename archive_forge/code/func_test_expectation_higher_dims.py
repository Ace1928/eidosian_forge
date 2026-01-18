import numpy as np
import pytest
import cirq
def test_expectation_higher_dims():
    qubit = cirq.NamedQid('q0', dimension=2)
    qutrit = cirq.NamedQid('q1', dimension=3)
    with pytest.raises(ValueError, match='Only qubits are supported'):
        cirq.ProjectorString({qutrit: 0})
    d = cirq.ProjectorString({qubit: 0})
    with pytest.raises(ValueError, match='Only qubits are supported'):
        _ = (d.expectation_from_state_vector(np.zeros(2 * 3), {qubit: 0, qutrit: 0}),)
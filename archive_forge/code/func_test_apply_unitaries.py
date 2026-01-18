import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_apply_unitaries():
    a, b, c = cirq.LineQubit.range(3)
    result = cirq.apply_unitaries(unitary_values=[cirq.H(a), cirq.CNOT(a, b), cirq.H(c).controlled_by(b)], qubits=[a, b, c])
    np.testing.assert_allclose(result.reshape(8), [np.sqrt(0.5), 0, 0, 0, 0, 0, 0.5, 0.5], atol=1e-08)
    result = cirq.apply_unitaries(unitary_values=[cirq.H(a), cirq.CNOT(a, b), cirq.H(c).controlled_by(b)], qubits=[a, c, b])
    np.testing.assert_allclose(result.reshape(8), [np.sqrt(0.5), 0, 0, 0, 0, 0.5, 0, 0.5], atol=1e-08)
    result = cirq.apply_unitaries(unitary_values=[cirq.H(a), cirq.CNOT(a, b), cirq.H(c).controlled_by(b)], qubits=[a, b, c], args=cirq.ApplyUnitaryArgs.default(num_qubits=3))
    np.testing.assert_allclose(result.reshape(8), [np.sqrt(0.5), 0, 0, 0, 0, 0, 0.5, 0.5], atol=1e-08)
    result = cirq.apply_unitaries(unitary_values=[], qubits=[])
    np.testing.assert_allclose(result, [1])
    result = cirq.apply_unitaries(unitary_values=[], qubits=[], default=None)
    np.testing.assert_allclose(result, [1])
    with pytest.raises(TypeError, match='non-unitary'):
        _ = cirq.apply_unitaries(unitary_values=[cirq.depolarize(0.5).on(a)], qubits=[a])
    assert cirq.apply_unitaries(unitary_values=[cirq.depolarize(0.5).on(a)], qubits=[a], default=None) is None
    assert cirq.apply_unitaries(unitary_values=[cirq.depolarize(0.5).on(a)], qubits=[a], default=1) == 1
    with pytest.raises(ValueError, match='len'):
        _ = cirq.apply_unitaries(unitary_values=[], qubits=[], args=cirq.ApplyUnitaryArgs.default(1))
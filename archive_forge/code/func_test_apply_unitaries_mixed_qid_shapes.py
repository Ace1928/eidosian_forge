import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_apply_unitaries_mixed_qid_shapes():

    class PlusOneMod3Gate(cirq.testing.SingleQubitGate):

        def _qid_shape_(self):
            return (3,)

        def _unitary_(self):
            return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    class PlusOneMod4Gate(cirq.testing.SingleQubitGate):

        def _qid_shape_(self):
            return (4,)

        def _unitary_(self):
            return np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    a, b = cirq.LineQid.for_qid_shape((3, 4))
    result = cirq.apply_unitaries(unitary_values=[PlusOneMod3Gate().on(a.with_dimension(3)), cirq.X(a.with_dimension(2)), cirq.CNOT(a.with_dimension(2), b.with_dimension(2)), cirq.CNOT(a.with_dimension(2), b.with_dimension(2)), cirq.X(a.with_dimension(2)), PlusOneMod3Gate().on(a.with_dimension(3)), PlusOneMod3Gate().on(a.with_dimension(3))], qubits=[a, b])
    np.testing.assert_allclose(result.reshape(12), [1] + [0] * 11, atol=1e-08)
    result = cirq.apply_unitaries(unitary_values=[PlusOneMod3Gate().on(a.with_dimension(3)), cirq.X(a.with_dimension(2)), cirq.CNOT(a.with_dimension(2), b.with_dimension(2)), cirq.CNOT(a.with_dimension(2), b.with_dimension(2)), cirq.X(a.with_dimension(2)), PlusOneMod3Gate().on(a.with_dimension(3)), PlusOneMod3Gate().on(a.with_dimension(3))], qubits=[a, b], args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((3, 4), dtype=np.complex64), available_buffer=cirq.eye_tensor((3, 4), dtype=np.complex64), axes=(0, 1)))
    np.testing.assert_allclose(result.reshape(12, 12), np.eye(12), atol=1e-08)
    result = cirq.apply_unitaries(unitary_values=[PlusOneMod3Gate().on(a.with_dimension(3)), cirq.X(a.with_dimension(2)), PlusOneMod4Gate().on(b.with_dimension(4)), PlusOneMod4Gate().on(b.with_dimension(4)), cirq.X(b.with_dimension(2)), PlusOneMod4Gate().on(b.with_dimension(4)), PlusOneMod4Gate().on(b.with_dimension(4)), cirq.CNOT(a.with_dimension(2), b.with_dimension(2)), PlusOneMod4Gate().on(b.with_dimension(4)), cirq.X(b.with_dimension(2)), cirq.CNOT(a.with_dimension(2), b.with_dimension(2)), cirq.X(a.with_dimension(2)), PlusOneMod3Gate().on(a.with_dimension(3)), PlusOneMod3Gate().on(a.with_dimension(3))], qubits=[a, b], args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((3, 4), dtype=np.complex64), available_buffer=cirq.eye_tensor((3, 4), dtype=np.complex64), axes=(0, 1)))
    np.testing.assert_allclose(result.reshape(12, 12), np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]), atol=1e-08)
import numpy as np
import pytest
import cirq
def test_apply_channel_channel_fallback_one_qubit_random_on_two_qubits():
    for _ in range(25):
        state = cirq.testing.random_superposition(4)
        rho = np.outer(np.conjugate(state), state)
        u = cirq.testing.random_unitary(2)
        full_u = np.kron(u, np.eye(2, dtype=np.complex128))
        expected = 0.5 * rho + 0.5 * np.dot(np.dot(full_u, rho), np.conjugate(np.transpose(full_u)))
        rho.shape = (2, 2, 2, 2)
        expected.shape = (2, 2, 2, 2)

        class HasChannel:

            def _kraus_(self):
                return (np.sqrt(0.5) * np.eye(2, dtype=np.complex128), np.sqrt(0.5) * u)
        result = apply_channel(HasChannel(), rho, [0], [2], assert_result_is_out_buf=True)
        np.testing.assert_almost_equal(result, expected)
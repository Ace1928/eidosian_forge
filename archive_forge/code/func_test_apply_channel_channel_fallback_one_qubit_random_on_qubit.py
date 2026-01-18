import numpy as np
import pytest
import cirq
def test_apply_channel_channel_fallback_one_qubit_random_on_qubit():
    for _ in range(25):
        state = cirq.testing.random_superposition(2)
        rho = np.outer(np.conjugate(state), state)
        u = cirq.testing.random_unitary(2)
        expected = 0.5 * rho + 0.5 * np.dot(np.dot(u, rho), np.conjugate(np.transpose(u)))

        class HasChannel:

            def _kraus_(self):
                return (np.sqrt(0.5) * np.eye(2, dtype=np.complex128), np.sqrt(0.5) * u)
        result = apply_channel(HasChannel(), rho, [0], [1], assert_result_is_out_buf=True)
        np.testing.assert_almost_equal(result, expected)
import numpy as np
import pytest
import cirq
def test_apply_channel_channel_fallback_simple():
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    class HasChannel:

        def _kraus_(self):
            return (np.sqrt(0.5) * np.eye(2, dtype=np.complex128), np.sqrt(0.5) * x)
    rho = np.copy(x)
    result = apply_channel(HasChannel(), rho, [0], [1], assert_result_is_out_buf=True)
    np.testing.assert_almost_equal(result, x)
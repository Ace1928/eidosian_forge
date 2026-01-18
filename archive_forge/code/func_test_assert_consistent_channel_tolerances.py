import pytest
import numpy as np
import cirq
def test_assert_consistent_channel_tolerances():
    channel = cirq.KrausChannel(kraus_ops=(np.array([[0, np.sqrt(1 - 1e-05)], [0, 0]]), np.array([[1, 0], [0, 0]])))
    cirq.testing.assert_consistent_channel(channel, rtol=1e-05, atol=0)
    with pytest.raises(AssertionError):
        cirq.testing.assert_consistent_channel(channel, rtol=1e-06, atol=0)
    cirq.testing.assert_consistent_channel(channel, rtol=0, atol=1e-05)
    with pytest.raises(AssertionError):
        cirq.testing.assert_consistent_channel(channel, rtol=0, atol=1e-06)
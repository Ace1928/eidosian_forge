import pytest
import numpy as np
import cirq
def test_assert_consistent_channel_valid():
    channel = cirq.KrausChannel(kraus_ops=(np.array([[0, 1], [0, 0]]), np.array([[1, 0], [0, 0]])))
    cirq.testing.assert_consistent_channel(channel)
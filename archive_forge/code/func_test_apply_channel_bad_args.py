import numpy as np
import pytest
import cirq
def test_apply_channel_bad_args():
    target = np.zeros((3,) + (1, 2, 3) + (3, 1, 2) + (3,))
    with pytest.raises(ValueError, match='Invalid target_tensor shape'):
        cirq.apply_channel(cirq.IdentityGate(3, (1, 2, 3)), cirq.ApplyChannelArgs(target, np.zeros_like(target), np.zeros_like(target), np.zeros_like(target), (1, 2, 3), (4, 5, 6)))
    target = np.zeros((2, 3, 2, 3))
    with pytest.raises(ValueError, match='Invalid channel qid shape'):
        cirq.apply_channel(cirq.IdentityGate(2, (2, 9)), cirq.ApplyChannelArgs(target, np.zeros_like(target), np.zeros_like(target), np.zeros_like(target), (0, 1), (2, 3)))
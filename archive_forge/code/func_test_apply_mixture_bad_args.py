from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def test_apply_mixture_bad_args():
    target = np.zeros((3,) + (1, 2, 3) + (3, 1, 2) + (3,))
    with pytest.raises(ValueError, match='Invalid target_tensor shape'):
        cirq.apply_mixture(cirq.IdentityGate(3, (1, 2, 3)), cirq.ApplyMixtureArgs(target, np.zeros_like(target), np.zeros_like(target), np.zeros_like(target), (1, 2, 3), (4, 5, 6)), default=np.array([]))
    target = np.zeros((2, 3, 2, 3))
    with pytest.raises(ValueError, match='Invalid mixture qid shape'):
        cirq.apply_mixture(cirq.IdentityGate(2, (2, 9)), cirq.ApplyMixtureArgs(target, np.zeros_like(target), np.zeros_like(target), np.zeros_like(target), (0, 1), (2, 3)), default=np.array([]))
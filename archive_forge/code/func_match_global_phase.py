import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def match_global_phase(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Phases the given matrices so that they agree on the phase of one entry.

    To maximize precision, the position with the largest entry from one of the
    matrices is used when attempting to compute the phase difference between
    the two matrices.

    Args:
        a: A numpy array.
        b: Another numpy array.

    Returns:
        A tuple (a', b') where a' == b' implies a == b*exp(i t) for some t.
    """
    if a.shape != b.shape or a.size == 0:
        return (np.copy(a), np.copy(b))
    k = max(np.ndindex(*a.shape), key=lambda t: abs(b[t]))

    def dephase(v):
        r = np.real(v)
        i = np.imag(v)
        if i == 0:
            return -1 if r < 0 else 1
        if r == 0:
            return 1j if i < 0 else -1j
        return np.exp(-1j * np.arctan2(i, r))
    return (a * dephase(a[k]), b * dephase(b[k]))
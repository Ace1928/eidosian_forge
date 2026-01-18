import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def recompose_so4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == (2, 2)
    assert b.shape == (2, 2)
    assert cirq.is_special_unitary(a)
    assert cirq.is_special_unitary(b)
    magic = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) * np.sqrt(0.5)
    result = np.real(cirq.dot(np.conj(magic.T), cirq.kron(a, b), magic))
    assert cirq.is_orthogonal(result)
    return result
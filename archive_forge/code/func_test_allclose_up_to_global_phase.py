import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_allclose_up_to_global_phase():
    assert cirq.allclose_up_to_global_phase(np.array([1]), np.array([1j]))
    assert not cirq.allclose_up_to_global_phase(np.array([[[1]]]), np.array([1]))
    assert cirq.allclose_up_to_global_phase(np.array([[1]]), np.array([[1]]))
    assert cirq.allclose_up_to_global_phase(np.array([[1]]), np.array([[-1]]))
    assert cirq.allclose_up_to_global_phase(np.array([[0]]), np.array([[0]]))
    assert cirq.allclose_up_to_global_phase(np.array([[1, 2]]), np.array([[1j, 2j]]))
    assert cirq.allclose_up_to_global_phase(np.array([[1, 2.0000000001]]), np.array([[1j, 2j]]))
    assert not cirq.allclose_up_to_global_phase(np.array([[1]]), np.array([[1, 0]]))
    assert not cirq.allclose_up_to_global_phase(np.array([[1]]), np.array([[2]]))
    assert not cirq.allclose_up_to_global_phase(np.array([[1]]), np.array([[2]]))
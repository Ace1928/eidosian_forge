import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('u', [cirq.testing.random_two_qubit_circuit_with_czs(3).unitary(), 1 / np.sqrt(2) * np.array([[(1 - 1j) * 2 / np.sqrt(5), 0, 0, (1 - 1j) * 1 / np.sqrt(5)], [0, 0, 1 - 1j, 0], [0, 1 - 1j, 0, 0], [-(1 - 1j) * 1 / np.sqrt(5), 0, 0, (1 - 1j) * 2 / np.sqrt(5)]], dtype=np.complex128)])
def test_extract_right_diag(u):
    assert cirq.num_cnots_required(u) == 3
    diag = cirq.linalg.extract_right_diag(u)
    assert cirq.is_diagonal(diag)
    assert cirq.num_cnots_required(u @ diag) == 2
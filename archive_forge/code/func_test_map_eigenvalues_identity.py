import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('matrix', [X, cirq.kron(X, X), cirq.kron(X, Y), cirq.kron(X, np.eye(2))])
def test_map_eigenvalues_identity(matrix):
    identity_mapped = cirq.map_eigenvalues(matrix, lambda e: e)
    assert np.allclose(matrix, identity_mapped)
import numpy as np
import pytest
import cirq
def test_fidelity_commuting_matrices():
    d1 = np.random.uniform(size=N)
    d1 /= np.sum(d1)
    d2 = np.random.uniform(size=N)
    d2 /= np.sum(d2)
    mat1 = cirq.density_matrix(U @ np.diag(d1) @ U.T.conj())
    mat2 = U @ np.diag(d2) @ U.T.conj()
    np.testing.assert_allclose(cirq.fidelity(mat1, mat2, qid_shape=(15,)), np.sum(np.sqrt(d1 * d2)) ** 2)
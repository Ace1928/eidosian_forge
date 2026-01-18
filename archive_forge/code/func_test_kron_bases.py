import itertools
import numpy as np
import pytest
import scipy.linalg
import cirq
@pytest.mark.parametrize('basis1, basis2, expected_kron_basis', ((PAULI_BASIS, PAULI_BASIS, {'II': np.eye(4), 'IX': scipy.linalg.block_diag(X, X), 'IY': scipy.linalg.block_diag(Y, Y), 'IZ': np.diag([1, -1, 1, -1]), 'XI': np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]), 'XX': np.rot90(np.eye(4)), 'XY': np.rot90(np.diag([1j, -1j, 1j, -1j])), 'XZ': np.array([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]]), 'YI': np.array([[0, 0, -1j, 0], [0, 0, 0, -1j], [1j, 0, 0, 0], [0, 1j, 0, 0]]), 'YX': np.rot90(np.diag([1j, 1j, -1j, -1j])), 'YY': np.rot90(np.diag([-1, 1, 1, -1])), 'YZ': np.array([[0, 0, -1j, 0], [0, 0, 0, 1j], [1j, 0, 0, 0], [0, -1j, 0, 0]]), 'ZI': np.diag([1, 1, -1, -1]), 'ZX': scipy.linalg.block_diag(X, -X), 'ZY': scipy.linalg.block_diag(Y, -Y), 'ZZ': np.diag([1, -1, -1, 1])}), (STANDARD_BASIS, STANDARD_BASIS, {'abcd'[2 * row_outer + col_outer] + 'abcd'[2 * row_inner + col_inner]: _one_hot_matrix(4, 2 * row_outer + row_inner, 2 * col_outer + col_inner) for row_outer in range(2) for row_inner in range(2) for col_outer in range(2) for col_inner in range(2)})))
def test_kron_bases(basis1, basis2, expected_kron_basis):
    kron_basis = cirq.kron_bases(basis1, basis2)
    assert len(kron_basis) == 16
    assert set(kron_basis.keys()) == set(expected_kron_basis.keys())
    for name in kron_basis.keys():
        assert np.all(kron_basis[name] == expected_kron_basis[name])
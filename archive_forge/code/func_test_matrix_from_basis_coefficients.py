import itertools
import numpy as np
import pytest
import scipy.linalg
import cirq
@pytest.mark.parametrize('expansion', ({'I': 1}, {'X': 1}, {'Y': 1}, {'Z': 1}, {'X': 1, 'Z': 1}, {'I': 0.5, 'X': 0.4, 'Y': 0.3, 'Z': 0.2}, {'I': 1, 'X': 2, 'Y': 3, 'Z': 4}))
def test_matrix_from_basis_coefficients(expansion):
    m = cirq.matrix_from_basis_coefficients(expansion, PAULI_BASIS)
    for name, coefficient in expansion.items():
        element = PAULI_BASIS[name]
        expected_coefficient = cirq.hilbert_schmidt_inner_product(m, element) / cirq.hilbert_schmidt_inner_product(element, element)
        assert np.isclose(coefficient, expected_coefficient)
import itertools
import numpy as np
import pytest
import scipy.linalg
import cirq
@pytest.mark.parametrize('basis,repeat', itertools.product((PAULI_BASIS, STANDARD_BASIS), range(5)))
def test_kron_bases_repeat_sanity_checks(basis, repeat):
    product_basis = cirq.kron_bases(basis, repeat=repeat)
    assert len(product_basis) == 4 ** repeat
    for name1, matrix1 in product_basis.items():
        for name2, matrix2 in product_basis.items():
            p = cirq.hilbert_schmidt_inner_product(matrix1, matrix2)
            if name1 != name2:
                assert p == 0
            else:
                assert abs(p) >= 1
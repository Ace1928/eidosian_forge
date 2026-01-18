import itertools
import numpy as np
import pytest
import cirq
import sympy
@pytest.mark.parametrize('cnt', [-1, 4, 10])
def test_invalid_required_sqrt_iswap_count(cnt):
    u = TWO_SQRT_ISWAP_UNITARIES[0]
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='required_sqrt_iswap_count'):
        cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u, required_sqrt_iswap_count=cnt)
import itertools
import numpy as np
import pytest
import cirq
import sympy
@pytest.mark.parametrize('u', ONE_SQRT_ISWAP_UNITARIES)
def test_decomp_optimal1(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u)
    assert_valid_decomp(u, ops)
    assert_specific_sqrt_iswap_count(ops, 1)
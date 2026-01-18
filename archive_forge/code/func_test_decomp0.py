import itertools
import numpy as np
import pytest
import cirq
import sympy
@pytest.mark.parametrize('u', ZERO_UNITARIES)
def test_decomp0(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u, required_sqrt_iswap_count=0)
    assert_valid_decomp(u, ops)
    assert_specific_sqrt_iswap_count(ops, 0)
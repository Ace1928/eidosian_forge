import itertools
import numpy as np
import pytest
import cirq
import sympy
@pytest.mark.parametrize('u', [ZERO_UNITARIES[0], ONE_SQRT_ISWAP_UNITARIES[0], TWO_SQRT_ISWAP_UNITARIES[0], THREE_SQRT_ISWAP_UNITARIES[0]])
def test_decomp_sqrt_iswap_inv(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u, use_sqrt_iswap_inv=True)
    assert_valid_decomp(u, ops, two_qubit_gate=cirq.SQRT_ISWAP_INV)
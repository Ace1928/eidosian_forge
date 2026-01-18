import itertools
import numpy as np
import pytest
import cirq
import sympy
def test_decomp3_invalid():
    u = cirq.unitary(cirq.X ** 0.2)
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='Input must correspond to a 4x4 unitary matrix'):
        cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, u, required_sqrt_iswap_count=3)
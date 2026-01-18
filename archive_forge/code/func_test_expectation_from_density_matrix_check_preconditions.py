import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_expectation_from_density_matrix_check_preconditions():
    q0, q1, _, q3 = cirq.LineQubit.range(4)
    psum = cirq.X(q0) + 2 * cirq.Y(q1) + 3 * cirq.Z(q3)
    q_map = {q0: 0, q1: 1, q3: 2}
    not_psd = np.zeros((8, 8), dtype=np.complex64)
    not_psd[0, 0] = 1.1
    not_psd[1, 1] = -0.1
    with pytest.raises(ValueError, match='semidefinite'):
        psum.expectation_from_density_matrix(not_psd, q_map)
    _ = psum.expectation_from_density_matrix(not_psd, q_map, check_preconditions=False)
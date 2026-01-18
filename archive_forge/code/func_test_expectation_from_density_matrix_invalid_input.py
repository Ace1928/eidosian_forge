import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_expectation_from_density_matrix_invalid_input():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    psum = cirq.X(q0) + 2 * cirq.Y(q1) + 3 * cirq.Z(q3)
    q_map = {q0: 0, q1: 1, q3: 2}
    wf = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64)
    rho = np.kron(wf.conjugate().T, wf).reshape((8, 8))
    im_psum = (1j + 1) * psum
    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        im_psum.expectation_from_density_matrix(rho, q_map)
    with pytest.raises(TypeError, match='dtype'):
        psum.expectation_from_density_matrix(0.5 * np.eye(2, dtype=int), q_map)
    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_density_matrix(rho, 'bad type')
    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_density_matrix(rho, {'bad key': 1})
    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_density_matrix(rho, {q0: 'bad value'})
    with pytest.raises(ValueError, match='complete'):
        psum.expectation_from_density_matrix(rho, {q0: 0})
    with pytest.raises(ValueError, match='complete'):
        psum.expectation_from_density_matrix(rho, {q0: 0, q2: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_density_matrix(rho, {q0: -1, q1: 1, q3: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_density_matrix(rho, {q0: 0, q1: 3, q3: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_density_matrix(rho, {q0: 0, q1: 0, q3: 2})
    with pytest.raises(ValueError, match='hermitian'):
        psum.expectation_from_density_matrix(1j * np.eye(8), q_map)
    with pytest.raises(ValueError, match='trace'):
        psum.expectation_from_density_matrix(np.eye(8, dtype=np.complex64), q_map)
    not_psd = np.zeros((8, 8), dtype=np.complex64)
    not_psd[0, 0] = 1.1
    not_psd[1, 1] = -0.1
    with pytest.raises(ValueError, match='semidefinite'):
        psum.expectation_from_density_matrix(not_psd, q_map)
    not_square = np.ones((8, 9), dtype=np.complex64)
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_density_matrix(not_square, q_map)
    bad_wf = np.zeros(128, dtype=np.complex64)
    bad_wf[0] = 1
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_density_matrix(bad_wf, q_map)
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_density_matrix(rho.reshape((8, 8, 1)), q_map)
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_density_matrix(rho.reshape(-1), q_map)
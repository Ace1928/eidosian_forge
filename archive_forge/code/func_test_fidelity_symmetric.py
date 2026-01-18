import numpy as np
import pytest
import cirq
def test_fidelity_symmetric():
    np.testing.assert_allclose(cirq.fidelity(VEC1, VEC2), cirq.fidelity(VEC2, VEC1))
    np.testing.assert_allclose(cirq.fidelity(VEC1, MAT1), cirq.fidelity(MAT1, VEC1))
    np.testing.assert_allclose(cirq.fidelity(cirq.density_matrix(MAT1), MAT2), cirq.fidelity(cirq.density_matrix(MAT2), MAT1))
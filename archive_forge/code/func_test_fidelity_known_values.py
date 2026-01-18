import numpy as np
import pytest
import cirq
def test_fidelity_known_values():
    vec1 = np.array([1, 1j, -1, -1j]) * 0.5
    vec2 = np.array([1, -1, 1, -1], dtype=np.complex128) * 0.5
    vec3 = np.array([1, 0, 0, 0], dtype=np.complex128)
    tensor1 = np.reshape(vec1, (2, 2))
    mat1 = cirq.density_matrix(np.outer(vec1, vec1.conj()))
    mat2 = cirq.density_matrix(np.outer(vec2, vec2.conj()))
    mat3 = 0.3 * mat1.density_matrix() + 0.7 * mat2.density_matrix()
    np.testing.assert_allclose(cirq.fidelity(vec1, vec1), 1)
    np.testing.assert_allclose(cirq.fidelity(vec2, vec2), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, vec3), 0.25)
    np.testing.assert_allclose(cirq.fidelity(vec1, tensor1), 1)
    np.testing.assert_allclose(cirq.fidelity(tensor1, vec1), 1)
    np.testing.assert_allclose(cirq.fidelity(mat1, mat1), 1)
    np.testing.assert_allclose(cirq.fidelity(mat2, mat2), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat1), 1)
    np.testing.assert_allclose(cirq.fidelity(mat2, vec2), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, vec2), 0)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat2), 0)
    np.testing.assert_allclose(cirq.fidelity(mat1, vec2), 0)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat3), 0.3)
    np.testing.assert_allclose(cirq.fidelity(tensor1, mat3), 0.3)
    np.testing.assert_allclose(cirq.fidelity(mat3, tensor1), 0.3)
    np.testing.assert_allclose(cirq.fidelity(mat3, vec2), 0.7)
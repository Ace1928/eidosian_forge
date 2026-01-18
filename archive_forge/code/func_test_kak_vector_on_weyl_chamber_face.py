import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kak_vector_on_weyl_chamber_face():
    theta_swap = np.linspace(0, np.pi / 4, 10)
    k_vecs = np.zeros((10, 3))
    k_vecs[:, (0, 1)] = theta_swap[:, np.newaxis]
    kwargs = dict(global_phase=1j, single_qubit_operations_before=(X, Y), single_qubit_operations_after=(Z, 1j * X))
    unitaries = np.array([cirq.unitary(cirq.KakDecomposition(interaction_coefficients=(t, t, 0), **kwargs)) for t in theta_swap])
    actual = cirq.kak_vector(unitaries)
    np.testing.assert_almost_equal(actual, k_vecs)
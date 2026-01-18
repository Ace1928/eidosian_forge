import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_axis_angle_decomposition_unitary():
    u = cirq.testing.random_unitary(2)
    u = cirq.unitary(cirq.T)
    a = cirq.axis_angle(u)
    np.testing.assert_allclose(u, cirq.unitary(a), atol=1e-08)
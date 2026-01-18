import numpy as np
import pytest
import sympy
from scipy import linalg
import cirq
def test_iswap_inv_unitary():
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(cirq.ISWAP_INV), np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]]), atol=1e-08)
import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_is_cptp_tolerance():
    rt2_ish = np.sqrt(0.5) - 0.01
    atol = 0.25
    assert cirq.is_cptp(kraus_ops=[np.array([[1, 0], [0, rt2_ish]]), np.array([[0, rt2_ish], [0, 0]])], atol=atol)
    assert not cirq.is_cptp(kraus_ops=[np.array([[1, 0], [0, rt2_ish]]), np.array([[0, rt2_ish], [0, 0]])], atol=1e-08)
import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_axis_angle_init():
    a = cirq.AxisAngleDecomposition(angle=1, axis=(0, 1, 0), global_phase=1j)
    assert a.angle == 1
    assert a.axis == (0, 1, 0)
    assert a.global_phase == 1j
    with pytest.raises(ValueError, match='normalize'):
        cirq.AxisAngleDecomposition(angle=1, axis=(0, 0.5, 0), global_phase=1)
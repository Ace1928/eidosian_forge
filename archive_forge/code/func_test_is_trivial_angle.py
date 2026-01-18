import cmath
import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
from cirq.testing import random_two_qubit_circuit_with_czs
@pytest.mark.parametrize('rad,expected', (lambda err, largeErr: [(np.pi / 4, True), (np.pi / 4 + err, True), (np.pi / 4 + largeErr, False), (np.pi / 4 - err, True), (np.pi / 4 - largeErr, False), (-np.pi / 4, True), (-np.pi / 4 + err, True), (-np.pi / 4 + largeErr, False), (-np.pi / 4 - err, True), (-np.pi / 4 - largeErr, False), (0, True), (err, True), (largeErr, False), (-err, True), (-largeErr, False), (np.pi / 8, False), (-np.pi / 8, False)])(1e-08 * 2 / 3, 1e-08 * 4 / 3))
def test_is_trivial_angle(rad, expected):
    tolerance = 1e-08
    out = _is_trivial_angle(rad, tolerance)
    assert out == expected, f'rad = {rad}'
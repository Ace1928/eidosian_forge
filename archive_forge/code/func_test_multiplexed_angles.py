from random import random
from typing import Callable
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag
import cirq
from cirq.transformers.analytical_decompositions.three_qubit_decomposition import (
def test_multiplexed_angles():
    theta = [random() * np.pi, random() * np.pi, random() * np.pi, random() * np.pi]
    angles = _multiplexed_angles(theta)
    assert np.isclose(theta[0], angles[0] + angles[1] + angles[2] + angles[3])
    assert np.isclose(theta[1], angles[0] + angles[1] - angles[2] - angles[3])
    assert np.isclose(theta[2], angles[0] - angles[1] - angles[2] + angles[3])
    assert np.isclose(theta[3], angles[0] - angles[1] + angles[2] - angles[3])
from random import random
from typing import Callable
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag
import cirq
from cirq.transformers.analytical_decompositions.three_qubit_decomposition import (
@pytest.mark.parametrize(['theta', 'num_czs'], [(np.array([0.5, 0.6, 0.7, 0.8]), 4), (np.array([0.0, 0.0, np.pi / 2, np.pi / 2]), 2), (np.zeros(4), 0), (np.repeat(np.pi / 4, repeats=4), 0), (np.array([0.5 * np.pi, -0.5 * np.pi, 0.7 * np.pi, -0.7 * np.pi]), 4), (np.array([0.3, -0.3, 0.3, -0.3]), 2), (np.array([0.3, 0.3, -0.3, -0.3]), 2)])
def test_cs_to_ops(theta, num_czs):
    a, b, c = cirq.LineQubit.range(3)
    cs = _theta_to_cs(theta)
    circuit_cs = cirq.Circuit(_cs_to_ops(a, b, c, theta))
    assert_almost_equal(circuit_cs.unitary(qubits_that_should_be_present=[a, b, c]), cs, 10)
    assert len([cz for cz in list(circuit_cs.all_operations()) if isinstance(cz.gate, cirq.CZPowGate)]) == num_czs, f'expected {num_czs} CZs got \n {circuit_cs} \n {circuit_cs.unitary()}'
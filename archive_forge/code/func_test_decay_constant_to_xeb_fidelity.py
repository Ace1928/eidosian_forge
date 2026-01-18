import numpy as np
import pytest
import cirq
from cirq.devices.noise_utils import (
@pytest.mark.parametrize('decay_constant,num_qubits,expected_output', [(0.01, 1, 1 - 0.99 * 1 / 2), (0.05, 2, 1 - 0.95 * 3 / 4)])
def test_decay_constant_to_xeb_fidelity(decay_constant, num_qubits, expected_output):
    val = decay_constant_to_xeb_fidelity(decay_constant, num_qubits)
    assert val == expected_output
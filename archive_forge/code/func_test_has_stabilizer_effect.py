from typing import cast
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_has_stabilizer_effect(num_qubits):
    assert cirq.has_stabilizer_effect(cirq.MeasurementGate(num_qubits, 'a'))
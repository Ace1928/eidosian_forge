import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
def test_find_measurements_invert_mask():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(q(0, 0), q(0, 1), q(0, 2), key='k', invert_mask=[False, True, True]))
    measurements = v2.find_measurements(circuit)
    assert len(measurements) == 1
    m = measurements[0]
    _check_measurement(m, 'k', [q(0, 0), q(0, 1), q(0, 2)], 1, [False, True, True])
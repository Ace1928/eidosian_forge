import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
def test_find_measurements_repeated_keys():
    circuit = cirq.Circuit(cirq.measure(q(0, 0), q(0, 1), key='k'), cirq.measure(q(0, 1), q(0, 2), key='j'), cirq.measure(q(0, 0), q(0, 1), key='k'))
    measurements = v2.find_measurements(circuit)
    assert len(measurements) == 2
    _check_measurement(measurements[0], 'k', [q(0, 0), q(0, 1)], 2)
    _check_measurement(measurements[1], 'j', [q(0, 1), q(0, 2)], 1)
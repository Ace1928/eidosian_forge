import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
def test_find_measurements_incompatible_repeated_keys():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(q(0, 0), q(0, 1), key='k'))
    circuit.append(cirq.measure(q(0, 1), q(0, 2), key='k'))
    with pytest.raises(ValueError, match='Incompatible repeated key'):
        v2.find_measurements(circuit)
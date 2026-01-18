import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('measurement_gate', (cirq.MeasurementGate(1, 'a'), cirq.PauliMeasurementGate([cirq.X], 'a')))
def test_fail_fast_measure(measurement_gate):
    assert not cirq.has_unitary(measurement_gate)
    qubit = cirq.NamedQubit('q0')
    circuit = cirq.Circuit()
    circuit += measurement_gate(qubit)
    circuit += cirq.H(qubit)
    assert not cirq.has_unitary(circuit)
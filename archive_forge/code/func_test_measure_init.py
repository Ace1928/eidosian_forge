from typing import cast
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_measure_init(num_qubits):
    assert cirq.MeasurementGate(num_qubits, 'a').num_qubits() == num_qubits
    assert cirq.MeasurementGate(num_qubits, key='a').key == 'a'
    assert cirq.MeasurementGate(num_qubits, key='a').mkey == cirq.MeasurementKey('a')
    assert cirq.MeasurementGate(num_qubits, key=cirq.MeasurementKey('a')).key == 'a'
    assert cirq.MeasurementGate(num_qubits, key=cirq.MeasurementKey('a')) == cirq.MeasurementGate(num_qubits, key='a')
    assert cirq.MeasurementGate(num_qubits, 'a', invert_mask=(True,)).invert_mask == (True,)
    assert cirq.qid_shape(cirq.MeasurementGate(num_qubits, 'a')) == (2,) * num_qubits
    cmap = {(0,): np.array([[0, 1], [1, 0]])}
    assert cirq.MeasurementGate(num_qubits, 'a', confusion_map=cmap).confusion_map == cmap
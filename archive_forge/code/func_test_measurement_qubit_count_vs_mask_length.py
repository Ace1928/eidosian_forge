from typing import cast
import numpy as np
import pytest
import cirq
def test_measurement_qubit_count_vs_mask_length():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    _ = cirq.MeasurementGate(num_qubits=1, key='a', invert_mask=(True,)).on(a)
    _ = cirq.MeasurementGate(num_qubits=2, key='a', invert_mask=(True, False)).on(a, b)
    _ = cirq.MeasurementGate(num_qubits=3, key='a', invert_mask=(True, False, True)).on(a, b, c)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(num_qubits=1, key='a', invert_mask=(True, False)).on(a)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(num_qubits=3, key='a', invert_mask=(True, False, True)).on(a, b)
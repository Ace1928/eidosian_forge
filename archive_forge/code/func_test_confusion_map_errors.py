import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_confusion_map_errors():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a', confusion_map={(0,): np.array([1])}), cirq.X(q1).with_classical_controls('a'))
    with pytest.raises(ValueError, match='map must be 2D'):
        _ = cirq.defer_measurements(circuit)
    circuit = cirq.Circuit(cirq.measure(q0, key='a', confusion_map={(0,): np.array([[0.7, 0.3]])}), cirq.X(q1).with_classical_controls('a'))
    with pytest.raises(ValueError, match='map must be square'):
        _ = cirq.defer_measurements(circuit)
    circuit = cirq.Circuit(cirq.measure(q0, key='a', confusion_map={(0,): np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])}), cirq.X(q1).with_classical_controls('a'))
    with pytest.raises(ValueError, match='size does not match'):
        _ = cirq.defer_measurements(circuit)
    circuit = cirq.Circuit(cirq.measure(q0, key='a', confusion_map={(0,): np.array([[-1, 2], [0, 1]])}), cirq.X(q1).with_classical_controls('a'))
    with pytest.raises(ValueError, match='negative probabilities'):
        _ = cirq.defer_measurements(circuit)
    circuit = cirq.Circuit(cirq.measure(q0, key='a', confusion_map={(0,): np.array([[0.3, 0.3], [0.3, 0.3]])}), cirq.X(q1).with_classical_controls('a'))
    with pytest.raises(ValueError, match='invalid probabilities'):
        _ = cirq.defer_measurements(circuit)
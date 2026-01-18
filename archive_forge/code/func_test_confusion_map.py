import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_confusion_map():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key='a', confusion_map={(0,): np.array([[0.8, 0.2], [0.1, 0.9]])}), cirq.X(q1).with_classical_controls('a'), cirq.measure(q1, key='b'))
    deferred = cirq.defer_measurements(circuit)
    sim = cirq.DensityMatrixSimulator()
    result = sim.sample(deferred, repetitions=10000)
    assert 5100 <= np.sum(result['a']) <= 5900
    assert np.all(result['a'] == result['b'])
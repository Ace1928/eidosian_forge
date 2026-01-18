import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_multi_qubit_control():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(q0, q1, key='a'), cirq.X(q2).with_classical_controls('a'), cirq.measure(q2, key='b'))
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma0 = _MeasurementQid('a', q0)
    q_ma1 = _MeasurementQid('a', q1)
    cirq.testing.assert_same_circuits(deferred, cirq.Circuit(cirq.CX(q0, q_ma0), cirq.CX(q1, q_ma1), cirq.X(q2).controlled_by(q_ma0, q_ma1, control_values=cirq.SumOfProducts(((0, 1), (1, 0), (1, 1)))), cirq.measure(q_ma0, q_ma1, key='a'), cirq.measure(q2, key='b')))
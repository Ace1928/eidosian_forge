import numpy as np
import pytest
import cirq
import sympy
def test_CCNOT():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit()
    circuit.append(cirq.CCNOT(q0, q1, q2))
    circuit.append(cirq.measure((q0, q1, q2), key='key'))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.CCNOT(q0, q1, q2))
    circuit.append(cirq.measure((q0, q1, q2), key='key'))
    circuit.append(cirq.X(q1))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.CCNOT(q0, q1, q2))
    circuit.append(cirq.measure((q0, q1, q2), key='key'))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.CCNOT(q0, q1, q2))
    circuit.append(cirq.measure((q0, q1, q2), key='key'))
    expected_results = {'key': np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)
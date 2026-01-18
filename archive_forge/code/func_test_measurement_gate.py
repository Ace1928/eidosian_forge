import numpy as np
import pytest
import cirq
import sympy
def test_measurement_gate():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.measure((q0, q1), key='key'))
    expected_results = {'key': np.array([[[0, 0]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)
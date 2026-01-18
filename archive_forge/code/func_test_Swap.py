import numpy as np
import pytest
import cirq
import sympy
def test_Swap():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(q0))
    circuit.append(cirq.SWAP(q0, q1))
    circuit.append(cirq.measure((q0, q1), key='key'))
    expected_results = {'key': np.array([[[0, 1]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)
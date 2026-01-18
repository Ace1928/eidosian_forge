import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_measurement_2qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.H(q2), cirq.measure(q0, q2))
    simulator = ccq.mps_simulator.MPSSimulator()
    repetitions = 1024
    measurement = simulator.run(circuit, repetitions=repetitions).measurements['q(0),q(2)']
    result_counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    for i in range(repetitions):
        key = str(measurement[i, 0]) + str(measurement[i, 1])
        result_counts[key] += 1
    for result_count in result_counts.values():
        assert result_count > repetitions * 0.15
        assert result_count < repetitions * 0.35
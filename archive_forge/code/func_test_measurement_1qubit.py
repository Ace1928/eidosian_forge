import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_measurement_1qubit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.H(q1), cirq.measure(q1))
    simulator = ccq.mps_simulator.MPSSimulator()
    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['q(1)'])[0] < 80
    assert sum(result.measurements['q(1)'])[0] > 20
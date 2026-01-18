import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_deterministic_gate_noise():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.I(q), cirq.measure(q))
    simulator1 = ccq.mps_simulator.MPSSimulator(noise=cirq.X)
    result1 = simulator1.run(circuit, repetitions=10)
    simulator2 = ccq.mps_simulator.MPSSimulator(noise=cirq.X)
    result2 = simulator2.run(circuit, repetitions=10)
    assert result1 == result2
    simulator3 = ccq.mps_simulator.MPSSimulator(noise=cirq.Z)
    result3 = simulator3.run(circuit, repetitions=10)
    assert result1 != result3
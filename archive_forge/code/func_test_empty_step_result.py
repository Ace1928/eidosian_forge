import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_empty_step_result():
    q0 = cirq.LineQubit(0)
    sim = ccq.mps_simulator.MPSSimulator()
    step_result = next(sim.simulate_moment_steps(cirq.Circuit(cirq.measure(q0))))
    assert 'TensorNetwork' in str(step_result)
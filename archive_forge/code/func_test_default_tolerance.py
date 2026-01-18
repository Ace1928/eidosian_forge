import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_default_tolerance():
    a, b = cirq.LineQubit.range(2)
    final_state_vector = cirq.Simulator().simulate(cirq.Circuit(cirq.H(a), cirq.H(b), cirq.CZ(a, b), cirq.measure(a))).final_state_vector.reshape((2, 2))
    cirq.sub_state_vector(final_state_vector, [0])
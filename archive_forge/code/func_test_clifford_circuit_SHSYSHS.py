import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_clifford_circuit_SHSYSHS():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.S(q0), cirq.H(q0), cirq.S(q0), cirq.Y(q0), cirq.S(q0), cirq.H(q0), cirq.S(q0), cirq.measure(q0))
    clifford_simulator = cirq.CliffordSimulator()
    state_vector_simulator = cirq.Simulator()
    np.testing.assert_almost_equal(clifford_simulator.simulate(circuit).final_state.state_vector(), state_vector_simulator.simulate(circuit).final_state_vector)
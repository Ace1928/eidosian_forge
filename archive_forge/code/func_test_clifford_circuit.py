import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('split', [True, False])
def test_clifford_circuit(split):
    q0, q1 = (cirq.LineQubit(0), cirq.LineQubit(1))
    circuit = cirq.Circuit()
    for _ in range(100):
        x = np.random.randint(7)
        if x == 0:
            circuit.append(cirq.X(np.random.choice((q0, q1))))
        elif x == 1:
            circuit.append(cirq.Z(np.random.choice((q0, q1))))
        elif x == 2:
            circuit.append(cirq.Y(np.random.choice((q0, q1))))
        elif x == 3:
            circuit.append(cirq.S(np.random.choice((q0, q1))))
        elif x == 4:
            circuit.append(cirq.H(np.random.choice((q0, q1))))
        elif x == 5:
            circuit.append(cirq.CNOT(q0, q1))
        elif x == 6:
            circuit.append(cirq.CZ(q0, q1))
    clifford_simulator = cirq.CliffordSimulator(split_untangled_states=split)
    state_vector_simulator = cirq.Simulator()
    np.testing.assert_almost_equal(clifford_simulator.simulate(circuit).final_state.state_vector(), state_vector_simulator.simulate(circuit).final_state_vector)
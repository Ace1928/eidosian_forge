import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('split', [True, False])
def test_clifford_circuit_3(split):
    q0, q1 = (cirq.LineQubit(0), cirq.LineQubit(1))
    circuit = cirq.Circuit()

    def random_clifford_gate():
        matrix = np.eye(2)
        for _ in range(10):
            matrix = matrix @ cirq.unitary(np.random.choice((cirq.H, cirq.S)))
        matrix *= np.exp(1j * np.random.uniform(0, 2 * np.pi))
        return cirq.MatrixGate(matrix)
    for _ in range(20):
        if np.random.randint(5) == 0:
            circuit.append(cirq.CNOT(q0, q1))
        else:
            circuit.append(random_clifford_gate()(np.random.choice((q0, q1))))
    clifford_simulator = cirq.CliffordSimulator(split_untangled_states=split)
    state_vector_simulator = cirq.Simulator()
    np.testing.assert_almost_equal(clifford_simulator.simulate(circuit).final_state.state_vector(), state_vector_simulator.simulate(circuit).final_state_vector, decimal=6)
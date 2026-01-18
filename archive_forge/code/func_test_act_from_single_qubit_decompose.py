import unittest.mock as mock
import numpy as np
import sympy
import cirq
def test_act_from_single_qubit_decompose():
    q0 = cirq.LineQubit(0)
    state = mock.Mock()
    args = cirq.StabilizerSimulationState(state=state, qubits=[q0])
    assert args._strat_act_from_single_qubit_decompose(cirq.MatrixGate(np.array([[0, 1], [1, 0]])), [q0]) is True
    state.apply_x.assert_called_with(0, 1.0, 0.0)
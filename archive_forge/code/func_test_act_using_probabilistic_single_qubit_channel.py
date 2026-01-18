from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def test_act_using_probabilistic_single_qubit_channel():

    class ProbabilisticSorX(cirq.Gate):

        def num_qubits(self) -> int:
            return 1

        def _kraus_(self):
            return [cirq.unitary(cirq.S) * np.sqrt(1 / 3), cirq.unitary(cirq.X) * np.sqrt(2 / 3)]
    initial_state = cirq.testing.random_superposition(dim=16).reshape((2,) * 4)
    mock_prng = mock.Mock()
    mock_prng.random.return_value = 1 / 3 + 1e-06
    args = cirq.StateVectorSimulationState(available_buffer=np.empty_like(initial_state), qubits=cirq.LineQubit.range(4), prng=mock_prng, initial_state=np.copy(initial_state), dtype=initial_state.dtype)
    cirq.act_on(ProbabilisticSorX(), args, [cirq.LineQubit(2)])
    np.testing.assert_allclose(args.target_tensor.reshape(16), cirq.final_state_vector(cirq.X(cirq.LineQubit(2)) ** (-1), initial_state=initial_state, qubit_order=cirq.LineQubit.range(4)), atol=1e-08)
    mock_prng.random.return_value = 1 / 3 - 1e-06
    args = cirq.StateVectorSimulationState(available_buffer=np.empty_like(initial_state), qubits=cirq.LineQubit.range(4), prng=mock_prng, initial_state=np.copy(initial_state), dtype=initial_state.dtype)
    cirq.act_on(ProbabilisticSorX(), args, [cirq.LineQubit(2)])
    np.testing.assert_allclose(args.target_tensor.reshape(16), cirq.final_state_vector(cirq.S(cirq.LineQubit(2)), initial_state=initial_state, qubit_order=cirq.LineQubit.range(4)), atol=1e-08)
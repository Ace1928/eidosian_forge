from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def test_act_using_adaptive_two_qubit_channel():

    class Decay11(cirq.Gate):

        def num_qubits(self) -> int:
            return 2

        def _kraus_(self):
            bottom_right = cirq.one_hot(index=(3, 3), shape=(4, 4), dtype=np.complex64)
            top_right = cirq.one_hot(index=(0, 3), shape=(4, 4), dtype=np.complex64)
            return [np.eye(4) * np.sqrt(3 / 4), (np.eye(4) - bottom_right) * np.sqrt(1 / 4), top_right * np.sqrt(1 / 4)]
    mock_prng = mock.Mock()

    def get_result(state: np.ndarray, sample: float):
        mock_prng.random.return_value = sample
        args = cirq.StateVectorSimulationState(available_buffer=np.empty_like(state), qubits=cirq.LineQubit.range(4), prng=mock_prng, initial_state=np.copy(state), dtype=cast(Type[np.complexfloating], state.dtype))
        cirq.act_on(Decay11(), args, [cirq.LineQubit(1), cirq.LineQubit(3)])
        return args.target_tensor

    def assert_not_affected(state: np.ndarray, sample: float):
        np.testing.assert_allclose(get_result(state, sample), state, atol=1e-08)
    all_zeroes = cirq.one_hot(index=(0, 0, 0, 0), shape=(2,) * 4, dtype=np.complex128)
    all_ones = cirq.one_hot(index=(1, 1, 1, 1), shape=(2,) * 4, dtype=np.complex128)
    decayed_all_ones = cirq.one_hot(index=(1, 0, 1, 0), shape=(2,) * 4, dtype=np.complex128)
    np.testing.assert_allclose(get_result(all_ones, 3 / 4 - 1e-08), all_ones)
    np.testing.assert_allclose(get_result(all_ones, 3 / 4 + 1e-08), decayed_all_ones)
    superpose = all_ones * np.sqrt(1 / 2) + all_zeroes * np.sqrt(1 / 2)
    np.testing.assert_allclose(get_result(superpose, 3 / 4 - 1e-08), superpose)
    np.testing.assert_allclose(get_result(superpose, 3 / 4 + 1e-08), all_zeroes)
    np.testing.assert_allclose(get_result(superpose, 7 / 8 - 1e-08), all_zeroes)
    np.testing.assert_allclose(get_result(superpose, 7 / 8 + 1e-08), decayed_all_ones)
    for _ in range(10):
        assert_not_affected(cirq.testing.random_superposition(dim=16).reshape((2,) * 4), sample=3 / 4 - 1e-08)
    for _ in range(10):
        mock_prng.random.return_value = 3 / 4 + 1e-06
        projected_state = cirq.testing.random_superposition(dim=16).reshape((2,) * 4)
        projected_state[cirq.slice_for_qubits_equal_to([1, 3], 3)] = 0
        projected_state /= np.linalg.norm(projected_state)
        assert abs(np.linalg.norm(projected_state) - 1) < 1e-08
        assert_not_affected(projected_state, sample=3 / 4 + 1e-08)
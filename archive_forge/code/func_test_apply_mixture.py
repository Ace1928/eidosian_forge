import unittest.mock as mock
import numpy as np
import sympy
import cirq
def test_apply_mixture():
    q0 = cirq.LineQubit(0)
    state = mock.Mock()
    args = cirq.StabilizerSimulationState(state=state, qubits=[q0])
    for _ in range(100):
        assert args._strat_apply_mixture(cirq.BitFlipChannel(0.5), [q0]) is True
    state.apply_x.assert_called_with(0, 1.0, 0.0)
    assert 10 < state.apply_x.call_count < 90
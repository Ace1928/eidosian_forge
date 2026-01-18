import numpy as np
import pytest
import cirq
import cirq.contrib.bayesian_network as ccb
@pytest.mark.parametrize('input_prob_q0,input_prob_q1,expected_prob_q2', [(0.0, 0.0, 0.1), (0.0, 1.0, 0.2), (1.0, 0.0, 0.3), (1.0, 1.0, 0.4)])
@pytest.mark.parametrize('decompose', [True, False])
def test_arc_probs(input_prob_q0, input_prob_q1, expected_prob_q2, decompose):
    q0, q1, q2 = cirq.LineQubit.range(3)
    gate = ccb.BayesianNetworkGate([('q0', input_prob_q0), ('q1', input_prob_q1), ('q2', None)], [('q2', ('q0', 'q1'), [0.1, 0.2, 0.3, 0.4])])
    if decompose:
        circuit = cirq.Circuit(cirq.decompose(gate.on(q0, q1, q2)))
    else:
        circuit = cirq.Circuit(gate.on(q0, q1, q2))
    result = cirq.Simulator().simulate(circuit, qubit_order=[q0, q1, q2], initial_state=0)
    probs = [abs(x) ** 2 for x in result.state_vector(copy=True)]
    actual_prob_q2_is_one = sum(probs[1::2])
    np.testing.assert_almost_equal(actual_prob_q2_is_one, expected_prob_q2, decimal=4)
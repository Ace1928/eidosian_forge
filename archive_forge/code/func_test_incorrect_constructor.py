import numpy as np
import pytest
import cirq
import cirq.contrib.bayesian_network as ccb
def test_incorrect_constructor():
    ccb.BayesianNetworkGate([('q0', 0.0), ('q1', None)], [('q1', ('q0',), [0.0, 0.0])])
    with pytest.raises(ValueError, match='Initial prob should be between 0 and 1.'):
        ccb.BayesianNetworkGate([('q0', 2016.0913), ('q1', None)], [('q1', ('q0',), [0.0, 0.0])])
    with pytest.raises(ValueError, match='Conditional prob params must be a tuple.'):
        ccb.BayesianNetworkGate([('q0', 0.0), ('q1', None)], [('q1', 'q0', [0.0, 0.0])])
    with pytest.raises(ValueError, match='Incorrect number of conditional probs.'):
        ccb.BayesianNetworkGate([('q0', 0.0), ('q1', None)], [('q1', ('q0',), [0.0])])
    with pytest.raises(ValueError, match='Conditional prob should be between 0 and 1.'):
        ccb.BayesianNetworkGate([('q0', 0.0), ('q1', None)], [('q1', ('q0',), [2016.0913, 0.0])])
import numpy as np
import pytest
import cirq
import cirq.contrib.bayesian_network as ccb
def test_basic_properties():
    gate = ccb.BayesianNetworkGate([('q0', None), ('q1', None), ('q2', None)], [])
    assert gate._has_unitary_()
    assert gate._qid_shape_() == (2, 2, 2)
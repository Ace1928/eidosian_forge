import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_depolarizing_mixture_two_qubits():
    d = cirq.depolarize(0.15, n_qubits=2)
    assert_mixtures_equal(cirq.mixture(d), ((0.85, np.eye(4)), (0.01, np.kron(np.eye(2), X)), (0.01, np.kron(np.eye(2), Y)), (0.01, np.kron(np.eye(2), Z)), (0.01, np.kron(X, np.eye(2))), (0.01, np.kron(X, X)), (0.01, np.kron(X, Y)), (0.01, np.kron(X, Z)), (0.01, np.kron(Y, np.eye(2))), (0.01, np.kron(Y, X)), (0.01, np.kron(Y, Y)), (0.01, np.kron(Y, Z)), (0.01, np.kron(Z, np.eye(2))), (0.01, np.kron(Z, X)), (0.01, np.kron(Z, Y)), (0.01, np.kron(Z, Z))))
    assert cirq.has_mixture(d)
import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit, initial_state', itertools.chain(itertools.product([cirq.Circuit(cirq.I(q0)), cirq.Circuit(cirq.X(q0)), cirq.Circuit(cirq.Y(q0)), cirq.Circuit(cirq.Z(q0)), cirq.Circuit(cirq.S(q0)), cirq.Circuit(cirq.T(q0))], density_operator_basis(n_qubits=1)), itertools.product([cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1)), cirq.Circuit(cirq.depolarize(0.2).on(q0), cirq.CNOT(q0, q1)), cirq.Circuit(cirq.X(q0), cirq.amplitude_damp(0.2).on(q0), cirq.depolarize(0.1).on(q1), cirq.CNOT(q0, q1))], density_operator_basis(n_qubits=2)), itertools.product([cirq.Circuit(cirq.depolarize(0.1, n_qubits=2).on(q0, q1), cirq.H(q2), cirq.CNOT(q1, q2), cirq.phase_damp(0.1).on(q0)), cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.TOFFOLI(q0, q1, q2))], density_operator_basis(n_qubits=3))))
def test_compare_circuits_superoperator_to_simulation(circuit, initial_state):
    """Compares action of circuit superoperator and circuit simulation."""
    assert circuit._has_superoperator_()
    superoperator = circuit._superoperator_()
    vectorized_initial_state = initial_state.reshape(-1)
    vectorized_final_state = superoperator @ vectorized_initial_state
    actual_state = np.reshape(vectorized_final_state, initial_state.shape)
    sim = cirq.DensityMatrixSimulator()
    expected_state = sim.simulate(circuit, initial_state=initial_state).final_density_matrix
    assert np.allclose(actual_state, expected_state)
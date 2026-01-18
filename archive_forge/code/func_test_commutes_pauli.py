import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('gate,pauli,half_turns', itertools.product(_all_clifford_gates(), _paulis, (1.0, 0.25, 0.5, -0.5)))
def test_commutes_pauli(gate, pauli, half_turns):
    pauli_gate = pauli if half_turns == 1 else pauli ** half_turns
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit(gate(q0), pauli_gate(q0)).unitary()
    mat_swap = cirq.Circuit(pauli_gate(q0), gate(q0)).unitary()
    commutes = cirq.commutes(gate, pauli_gate)
    commutes_check = np.allclose(mat, mat_swap)
    assert commutes == commutes_check, f'gate: {gate}, pauli {pauli}'
import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('gate,other', itertools.combinations(_all_clifford_gates(), r=2))
def test_commutes_single_qubit_gate(gate, other):
    q0 = cirq.NamedQubit('q0')
    gate_op = gate(q0)
    other_op = other(q0)
    mat = cirq.Circuit(gate_op, other_op).unitary()
    mat_swap = cirq.Circuit(other_op, gate_op).unitary()
    commutes = cirq.commutes(gate, other)
    commutes_check = cirq.allclose_up_to_global_phase(mat, mat_swap)
    assert commutes == commutes_check
    mat_swap = cirq.Circuit(gate.equivalent_gate_before(other)(q0), gate_op).unitary()
    assert_allclose_up_to_global_phase(mat, mat_swap, rtol=1e-07, atol=1e-07)
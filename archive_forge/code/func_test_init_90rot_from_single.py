import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('trans,frm', ((trans, frm) for trans, frm in itertools.product(_all_rotations(), _paulis) if trans[0] != frm))
def test_init_90rot_from_single(trans, frm):
    gate = cirq.SingleQubitCliffordGate.from_single_map({frm: trans})
    assert gate.pauli_tuple(frm) == trans
    _assert_not_mirror(gate)
    _assert_no_collision(gate)
    assert len(gate.decompose_rotation()) == 1
    assert gate.merged_with(gate).merged_with(gate).merged_with(gate) == cirq.SingleQubitCliffordGate.I
    trans_rev = (trans[0], not trans[1])
    gate_rev = cirq.SingleQubitCliffordGate.from_single_map({frm: trans_rev})
    assert gate ** (-1) == gate_rev
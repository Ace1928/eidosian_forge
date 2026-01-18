import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('trans1,trans2,from1', ((trans1, trans2, from1) for trans1, trans2, from1 in itertools.product(_all_rotations(), _all_rotations(), _paulis) if trans1[0] != trans2[0]))
def test_init_from_double_map_vs_kwargs(trans1, trans2, from1):
    from2 = cirq.Pauli.by_relative_index(from1, 1)
    from1_str, from2_str = (str(frm).lower() + '_to' for frm in (from1, from2))
    gate_kw = cirq.SingleQubitCliffordGate.from_double_map(**{from1_str: trans1, from2_str: trans2})
    gate_map = cirq.SingleQubitCliffordGate.from_double_map({from1: trans1, from2: trans2})
    assert gate_kw == gate_map
    assert gate_map.pauli_tuple(from1) == trans1
    assert gate_map.pauli_tuple(from2) == trans2
    _assert_not_mirror(gate_map)
    _assert_no_collision(gate_map)
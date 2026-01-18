import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('clifford_gate', cirq.SingleQubitCliffordGate.all_single_qubit_cliffords)
def test_from_unitary(clifford_gate):
    u = cirq.unitary(clifford_gate)
    result_gate = cirq.SingleQubitCliffordGate.from_unitary(u)
    assert result_gate == clifford_gate
    result_gate2, global_phase = cirq.SingleQubitCliffordGate.from_unitary_with_global_phase(u)
    assert result_gate2 == result_gate
    assert np.allclose(cirq.unitary(result_gate2) * global_phase, u)
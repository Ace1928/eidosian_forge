import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_from_unitary_not_clifford():
    u = cirq.unitary(cirq.CNOT)
    assert cirq.SingleQubitCliffordGate.from_unitary(u) is None
    assert cirq.SingleQubitCliffordGate.from_unitary_with_global_phase(u) is None
    u = 2 * cirq.unitary(cirq.X)
    assert cirq.SingleQubitCliffordGate.from_unitary(u) is None
    assert cirq.SingleQubitCliffordGate.from_unitary_with_global_phase(u) is None
    u = cirq.unitary(cirq.T)
    assert cirq.SingleQubitCliffordGate.from_unitary(u) is None
    assert cirq.SingleQubitCliffordGate.from_unitary_with_global_phase(u) is None
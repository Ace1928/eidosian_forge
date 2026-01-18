import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_from_unitary_with_phase_shift():
    u = np.exp(0.42j) * cirq.unitary(cirq.SingleQubitCliffordGate.Y_sqrt)
    gate = cirq.SingleQubitCliffordGate.from_unitary(u)
    assert gate == cirq.SingleQubitCliffordGate.Y_sqrt
    gate2, global_phase = cirq.SingleQubitCliffordGate.from_unitary_with_global_phase(u)
    assert gate2 == gate
    assert np.allclose(cirq.unitary(gate2) * global_phase, u)
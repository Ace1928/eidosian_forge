import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('gate,gate_equiv', ((cirq.SingleQubitCliffordGate.I, cirq.X ** 0), (cirq.SingleQubitCliffordGate.H, cirq.H), (cirq.SingleQubitCliffordGate.X, cirq.X), (cirq.SingleQubitCliffordGate.Y, cirq.Y), (cirq.SingleQubitCliffordGate.Z, cirq.Z), (cirq.SingleQubitCliffordGate.X_sqrt, cirq.X ** 0.5), (cirq.SingleQubitCliffordGate.X_nsqrt, cirq.X ** (-0.5)), (cirq.SingleQubitCliffordGate.Y_sqrt, cirq.Y ** 0.5), (cirq.SingleQubitCliffordGate.Y_nsqrt, cirq.Y ** (-0.5)), (cirq.SingleQubitCliffordGate.Z_sqrt, cirq.Z ** 0.5), (cirq.SingleQubitCliffordGate.Z_nsqrt, cirq.Z ** (-0.5))))
def test_known_matrix(gate, gate_equiv):
    assert cirq.has_unitary(gate)
    mat = cirq.unitary(gate)
    mat_check = cirq.unitary(gate_equiv)
    assert_allclose_up_to_global_phase(mat, mat_check, rtol=1e-07, atol=1e-07)
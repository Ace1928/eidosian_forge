import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('pauli,sqrt,expected', ((cirq.X, False, cirq.SingleQubitCliffordGate.X), (cirq.Y, False, cirq.SingleQubitCliffordGate.Y), (cirq.Z, False, cirq.SingleQubitCliffordGate.Z), (cirq.X, True, cirq.SingleQubitCliffordGate.X_sqrt), (cirq.Y, True, cirq.SingleQubitCliffordGate.Y_sqrt), (cirq.Z, True, cirq.SingleQubitCliffordGate.Z_sqrt)))
def test_init_from_pauli(pauli, sqrt, expected):
    gate = cirq.SingleQubitCliffordGate.from_pauli(pauli, sqrt=sqrt)
    assert gate == expected
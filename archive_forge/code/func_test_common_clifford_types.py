import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('name, expected_cls', [('I', cirq.SingleQubitCliffordGate), ('H', cirq.SingleQubitCliffordGate), ('X', cirq.SingleQubitCliffordGate), ('Y', cirq.SingleQubitCliffordGate), ('Z', cirq.SingleQubitCliffordGate), ('S', cirq.SingleQubitCliffordGate), ('X_sqrt', cirq.SingleQubitCliffordGate), ('X_nsqrt', cirq.SingleQubitCliffordGate), ('Y_sqrt', cirq.SingleQubitCliffordGate), ('Y_nsqrt', cirq.SingleQubitCliffordGate), ('Z_sqrt', cirq.SingleQubitCliffordGate), ('Z_nsqrt', cirq.SingleQubitCliffordGate), ('CNOT', cirq.CliffordGate), ('CZ', cirq.CliffordGate), ('SWAP', cirq.CliffordGate)])
def test_common_clifford_types(name: str, expected_cls: Type) -> None:
    assert isinstance(getattr(cirq.CliffordGate, name), expected_cls)
    assert isinstance(getattr(cirq.SingleQubitCliffordGate, name), expected_cls)
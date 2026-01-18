import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_multi_qubit_clifford_pow():
    assert cirq.CliffordGate.X ** (-1) == cirq.CliffordGate.X
    assert cirq.CliffordGate.H ** (-1) == cirq.CliffordGate.H
    assert cirq.CliffordGate.S ** 2 == cirq.CliffordGate.Z
    assert cirq.CliffordGate.S ** (-1) == cirq.CliffordGate.S ** 3
    assert cirq.CliffordGate.S ** (-3) == cirq.CliffordGate.S
    assert cirq.CliffordGate.CNOT ** 3 == cirq.CliffordGate.CNOT
    assert cirq.CliffordGate.CNOT ** (-3) == cirq.CliffordGate.CNOT
    with pytest.raises(TypeError):
        _ = cirq.CliffordGate.Z ** 0.25
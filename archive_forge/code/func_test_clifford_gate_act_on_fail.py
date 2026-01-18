import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_clifford_gate_act_on_fail():
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(cirq.CliffordGate.X, ExampleSimulationState(), qubits=())
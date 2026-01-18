import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_clifford_gate_from_tableau():
    t = cirq.CliffordGate.X.clifford_tableau
    assert cirq.CliffordGate.from_clifford_tableau(t) == cirq.CliffordGate.X
    t = cirq.CliffordGate.H.clifford_tableau
    assert cirq.CliffordGate.from_clifford_tableau(t) == cirq.CliffordGate.H
    t = cirq.CliffordGate.CNOT.clifford_tableau
    assert cirq.CliffordGate.from_clifford_tableau(t) == cirq.CliffordGate.CNOT
    with pytest.raises(ValueError, match='Input argument has to be a CliffordTableau instance.'):
        cirq.SingleQubitCliffordGate.from_clifford_tableau(123)
    with pytest.raises(ValueError, match='The number of qubit of input tableau should be 1'):
        t = cirq.CliffordTableau(num_qubits=2)
        cirq.SingleQubitCliffordGate.from_clifford_tableau(t)
    with pytest.raises(ValueError):
        t = cirq.CliffordTableau(num_qubits=1)
        t.xs = np.array([1, 1]).reshape(2, 1)
        t.zs = np.array([1, 1]).reshape(2, 1)
        cirq.CliffordGate.from_clifford_tableau(t)
    with pytest.raises(ValueError, match='Input argument has to be a CliffordTableau instance.'):
        cirq.CliffordGate.from_clifford_tableau(1)
import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_all_single_qubit_clifford_unitaries():
    i = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    cs = [cirq.unitary(c) for c in cirq.CliffordGate.all_single_qubit_cliffords]
    assert cirq.equal_up_to_global_phase(cs[0], i)
    assert cirq.equal_up_to_global_phase(cs[1], x)
    assert cirq.equal_up_to_global_phase(cs[2], y)
    assert cirq.equal_up_to_global_phase(cs[3], z)
    assert cirq.equal_up_to_global_phase(cs[4], (i - 1j * x) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[5], (i - 1j * y) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[6], (i - 1j * z) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[7], (i + 1j * x) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[8], (i + 1j * y) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[9], (i + 1j * z) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[10], (z + x) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[11], (x + y) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[12], (y + z) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[13], (z - x) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[14], (x - y) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[15], (y - z) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[16], (i - 1j * (x + y + z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[17], (i - 1j * (x + y - z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[18], (i - 1j * (x - y + z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[19], (i - 1j * (x - y - z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[20], (i - 1j * (-x + y + z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[21], (i - 1j * (-x + y - z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[22], (i - 1j * (-x - y + z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[23], (i - 1j * (-x - y - z)) / 2)
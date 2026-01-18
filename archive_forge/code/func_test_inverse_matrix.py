import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_inverse_matrix(gate):
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit(gate(q0)).unitary()
    mat_inv = cirq.Circuit(cirq.inverse(gate)(q0)).unitary()
    assert_allclose_up_to_global_phase(mat, mat_inv.T.conj(), rtol=1e-07, atol=1e-07)
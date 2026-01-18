import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_identity_unitary(num_qubits):
    i = cirq.IdentityGate(num_qubits)
    assert np.allclose(cirq.unitary(i), np.identity(2 ** num_qubits))
    i3 = cirq.IdentityGate(num_qubits, (3,) * num_qubits)
    assert np.allclose(cirq.unitary(i3), np.identity(3 ** num_qubits))
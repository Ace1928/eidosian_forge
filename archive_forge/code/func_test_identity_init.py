import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_identity_init(num_qubits):
    assert cirq.IdentityGate(num_qubits).num_qubits() == num_qubits
    assert cirq.qid_shape(cirq.IdentityGate(num_qubits)) == (2,) * num_qubits
    assert cirq.qid_shape(cirq.IdentityGate(3, (1, 2, 3))) == (1, 2, 3)
    assert cirq.qid_shape(cirq.IdentityGate(qid_shape=(1, 2, 3))) == (1, 2, 3)
    with pytest.raises(ValueError, match='len.* !='):
        cirq.IdentityGate(5, qid_shape=(1, 2))
    with pytest.raises(ValueError, match='Specify either'):
        cirq.IdentityGate()
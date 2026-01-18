import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_apply_unitary():
    v = np.array([1, 0])
    result = cirq.apply_unitary(cirq.I, cirq.ApplyUnitaryArgs(v, np.array([0, 1]), (0,)))
    assert result is v
    v = np.array([1, 0, 0])
    result = cirq.apply_unitary(cirq.IdentityGate(1, (3,)), cirq.ApplyUnitaryArgs(v, np.array([0, 1, 2]), (0,)))
    assert result is v
import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('op', (cirq.X(q0), cirq.Y(q1), cirq.XX(q0, q1), cirq.CZ(q0, q1), cirq.FREDKIN(q0, q1, q2), cirq.ControlledOperation((q0, q1), cirq.H(q2)), cirq.ParallelGate(cirq.X, 3).on(q0, q1, q2), cirq.PauliString({q0: cirq.X, q1: cirq.Y, q2: cirq.Z})))
def test_empty_linear_combination_of_operations_accepts_all_operations(op):
    combination = cirq.LinearCombinationOfOperations({})
    combination[op] = -0.5j
    assert len(combination) == 1
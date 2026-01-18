import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_mapped_circuit_preserves_moments():
    q0, q1 = cirq.LineQubit.range(2)
    fc = cirq.FrozenCircuit(cirq.Moment(cirq.X(q0)), cirq.Moment(cirq.X(q1)))
    op = cirq.CircuitOperation(fc)
    assert op.mapped_circuit() == fc
    assert op.repeat(3).mapped_circuit(deep=True) == fc * 3
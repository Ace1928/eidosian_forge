import numpy as np
import pytest
import sympy
import cirq
def test_gate_op_repr():
    gate = cirq.GlobalPhaseGate(1j)
    cirq.testing.assert_equivalent_repr(gate.on())
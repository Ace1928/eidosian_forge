import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_gate_operation_init():
    q = cirq.NamedQubit('q')
    g = cirq.testing.SingleQubitGate()
    v = cirq.GateOperation(g, (q,))
    assert v.gate == g
    assert v.qubits == (q,)
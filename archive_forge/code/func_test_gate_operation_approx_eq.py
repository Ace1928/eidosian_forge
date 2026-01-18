import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_gate_operation_approx_eq():
    a = [cirq.NamedQubit('r1')]
    b = [cirq.NamedQubit('r2')]
    assert cirq.approx_eq(cirq.GateOperation(cirq.XPowGate(), a), cirq.GateOperation(cirq.XPowGate(), a))
    assert not cirq.approx_eq(cirq.GateOperation(cirq.XPowGate(), a), cirq.GateOperation(cirq.XPowGate(), b))
    assert cirq.approx_eq(cirq.GateOperation(cirq.XPowGate(exponent=0), a), cirq.GateOperation(cirq.XPowGate(exponent=1e-09), a))
    assert not cirq.approx_eq(cirq.GateOperation(cirq.XPowGate(exponent=0), a), cirq.GateOperation(cirq.XPowGate(exponent=1e-07), a))
    assert cirq.approx_eq(cirq.GateOperation(cirq.XPowGate(exponent=0), a), cirq.GateOperation(cirq.XPowGate(exponent=1e-07), a), atol=1e-06)
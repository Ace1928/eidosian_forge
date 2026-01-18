import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_is_native_xmon_gate():
    assert programs.is_native_xmon_gate(cirq.CZ)
    assert programs.is_native_xmon_gate(cirq.X ** 0.5)
    assert programs.is_native_xmon_gate(cirq.Y ** 0.5)
    assert programs.is_native_xmon_gate(cirq.Z ** 0.5)
    assert programs.is_native_xmon_gate(cirq.PhasedXPowGate(phase_exponent=0.2) ** 0.5)
    assert programs.is_native_xmon_gate(cirq.Z ** 1)
    assert not programs.is_native_xmon_gate(cirq.CCZ)
    assert not programs.is_native_xmon_gate(cirq.SWAP)
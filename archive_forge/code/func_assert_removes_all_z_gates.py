import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
def assert_removes_all_z_gates(circuit: cirq.Circuit, eject_parameterized: bool=True):
    optimized = cirq.eject_z(circuit, eject_parameterized=eject_parameterized)
    for op in optimized.all_operations():
        if isinstance(op.gate, cirq.PhasedXZGate) and (eject_parameterized or not cirq.is_parameterized(op.gate.z_exponent)):
            assert op.gate.z_exponent == 0
    if cirq.is_parameterized(circuit):
        for a in (0, 0.1, 0.5, 1.0, -1.0, 3.0):
            cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.resolve_parameters(circuit, {'a': a}), cirq.resolve_parameters(optimized, {'a': a}), atol=1e-08)
    else:
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, optimized, atol=1e-08)
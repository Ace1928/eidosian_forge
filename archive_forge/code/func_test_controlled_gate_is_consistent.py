from typing import Union, Tuple, cast
import numpy as np
import pytest
import sympy
import cirq
from cirq.type_workarounds import NotImplementedType
@pytest.mark.parametrize('gate, should_decompose_to_target', [(cirq.X, True), (cirq.X ** 0.5, True), (cirq.rx(np.pi), True), (cirq.rx(np.pi / 2), True), (cirq.Z, True), (cirq.H, True), (cirq.CNOT, True), (cirq.SWAP, True), (cirq.CCZ, True), (cirq.ControlledGate(cirq.ControlledGate(cirq.CCZ)), True), (GateUsingWorkspaceForApplyUnitary(), True), (GateAllocatingNewSpaceForResult(), True), (cirq.IdentityGate(qid_shape=(3, 4)), True), (cirq.ControlledGate(cirq.XXPowGate(exponent=0.25, global_shift=-0.5), num_controls=2, control_values=(1, (1, 0))), True), (cirq.MatrixGate(np.kron(*(cirq.unitary(cirq.H),) * 2), qid_shape=(4,)), False), (cirq.MatrixGate(cirq.testing.random_unitary(4, random_state=1234)), False), (cirq.XX ** sympy.Symbol('s'), True), (cirq.CZ ** sympy.Symbol('s'), True), (C_01_10_11H, False), (C_xorH, False), (C0C_xorH, False)])
def test_controlled_gate_is_consistent(gate: cirq.Gate, should_decompose_to_target):
    cgate = cirq.ControlledGate(gate)
    cirq.testing.assert_implements_consistent_protocols(cgate)
    cirq.testing.assert_decompose_ends_at_default_gateset(cgate, ignore_known_gates=not should_decompose_to_target)
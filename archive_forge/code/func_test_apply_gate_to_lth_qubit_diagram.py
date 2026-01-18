import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_apply_gate_to_lth_qubit_diagram():
    gate = cirq_ft.ApplyGateToLthQubit(cirq_ft.SelectionRegister('selection', 3, 5), lambda n: cirq.Z if n & 1 else cirq.I, control_regs=cirq_ft.Signature.build(control=2))
    circuit = cirq.Circuit(gate.on_registers(**infra.get_named_qubits(gate.signature)))
    qubits = list((q for v in infra.get_named_qubits(gate.signature).values() for q in v))
    cirq.testing.assert_has_diagram(circuit, '\ncontrol0: ─────@────\n               │\ncontrol1: ─────@────\n               │\nselection0: ───In───\n               │\nselection1: ───In───\n               │\nselection2: ───In───\n               │\ntarget0: ──────I────\n               │\ntarget1: ──────Z────\n               │\ntarget2: ──────I────\n               │\ntarget3: ──────Z────\n               │\ntarget4: ──────I────\n', qubit_order=qubits)
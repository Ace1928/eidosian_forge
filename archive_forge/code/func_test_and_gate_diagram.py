import itertools
import random
from typing import List, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_and_gate_diagram():
    gate = cirq_ft.And((1, 0, 1, 0, 1, 0))
    qubit_regs = infra.get_named_qubits(gate.signature)
    op = gate.on_registers(**qubit_regs)
    ctrl, junk, target = (qubit_regs['ctrl'].flatten(), qubit_regs['junk'].flatten(), qubit_regs['target'].flatten())
    c_and_a = sum(zip(ctrl[1:], junk), ()) + (ctrl[-1],)
    qubit_order = np.concatenate([ctrl[0:1], c_and_a, target])
    cirq.testing.assert_has_diagram(cirq.Circuit(op), '\nctrl[0]: ───@─────\n            │\nctrl[1]: ───(0)───\n            │\njunk[0]: ───Anc───\n            │\nctrl[2]: ───@─────\n            │\njunk[1]: ───Anc───\n            │\nctrl[3]: ───(0)───\n            │\njunk[2]: ───Anc───\n            │\nctrl[4]: ───@─────\n            │\njunk[3]: ───Anc───\n            │\nctrl[5]: ───(0)───\n            │\ntarget: ────And───\n', qubit_order=qubit_order)
    cirq.testing.assert_has_diagram(cirq.Circuit(op ** (-1)), '\nctrl[0]: ───@──────\n            │\nctrl[1]: ───(0)────\n            │\njunk[0]: ───Anc────\n            │\nctrl[2]: ───@──────\n            │\njunk[1]: ───Anc────\n            │\nctrl[3]: ───(0)────\n            │\njunk[2]: ───Anc────\n            │\nctrl[4]: ───@──────\n            │\njunk[3]: ───Anc────\n            │\nctrl[5]: ───(0)────\n            │\ntarget: ────And†───\n', qubit_order=qubit_order)
    decomposed_circuit = cirq.Circuit(cirq.decompose_once(op)) + cirq.Circuit(cirq.decompose_once(op ** (-1)))
    cirq.testing.assert_has_diagram(decomposed_circuit, '\nctrl[0]: ───@─────────────────────────────────────────────────────────@──────\n            │                                                         │\nctrl[1]: ───(0)───────────────────────────────────────────────────────(0)────\n            │                                                         │\njunk[0]: ───And───@────────────────────────────────────────────@──────And†───\n                  │                                            │\nctrl[2]: ─────────@────────────────────────────────────────────@─────────────\n                  │                                            │\njunk[1]: ─────────And───@───────────────────────────────@──────And†──────────\n                        │                               │\nctrl[3]: ───────────────(0)─────────────────────────────(0)──────────────────\n                        │                               │\njunk[2]: ───────────────And───@──────────────────@──────And†─────────────────\n                              │                  │\nctrl[4]: ─────────────────────@──────────────────@───────────────────────────\n                              │                  │\njunk[3]: ─────────────────────And───@─────@──────And†────────────────────────\n                                    │     │\nctrl[5]: ───────────────────────────(0)───(0)────────────────────────────────\n                                    │     │\ntarget: ────────────────────────────And───And†───────────────────────────────\n', qubit_order=qubit_order)
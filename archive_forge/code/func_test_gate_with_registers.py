import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.infra import split_qubits, merge_qubits, get_named_qubits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_gate_with_registers():
    tg = _TestGate()
    assert tg._num_qubits_() == 8
    qubits = cirq.LineQubit.range(8)
    circ = cirq.Circuit(tg._decompose_(qubits))
    assert circ.operation_at(cirq.LineQubit(3), 0).gate == cirq.H
    op1 = tg.on_registers(r1=qubits[:5], r2=qubits[6:], r3=qubits[5])
    op2 = tg.on(*qubits[:5], *qubits[6:], qubits[5])
    assert op1 == op2
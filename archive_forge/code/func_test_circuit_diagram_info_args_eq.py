import numpy as np
import pytest
import sympy
import cirq
def test_circuit_diagram_info_args_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT)
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(known_qubits=None, known_qubit_count=None, use_unicode_characters=False, precision=None, label_map=None))
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(known_qubits=None, known_qubit_count=None, use_unicode_characters=True, precision=None, label_map=None))
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(known_qubits=cirq.LineQubit.range(3), known_qubit_count=3, use_unicode_characters=False, precision=None, label_map=None))
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(known_qubits=cirq.LineQubit.range(2), known_qubit_count=2, use_unicode_characters=False, precision=None, label_map=None))
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(known_qubits=cirq.LineQubit.range(2), known_qubit_count=2, use_unicode_characters=False, precision=None, label_map=None, include_tags=False))
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(known_qubits=cirq.LineQubit.range(2), known_qubit_count=2, use_unicode_characters=False, precision=None, label_map={cirq.LineQubit(0): 5, cirq.LineQubit(1): 7}))
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(known_qubits=cirq.LineQubit.range(2), known_qubit_count=2, use_unicode_characters=False, precision=None, label_map={cirq.LineQubit(0): 5, cirq.LineQubit(1): 7}, transpose=True))
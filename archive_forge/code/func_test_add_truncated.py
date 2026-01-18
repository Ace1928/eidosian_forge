import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_add_truncated():
    num_bits = 3
    num_anc = num_bits - 1
    gate = cirq_ft.AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits)))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    initial_state = [0, 0, 1, 0, 0, 1, 0, 0]
    final_state = [0, 0, 1, 0, 0, 0, 0, 0]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)
    num_bits = 4
    num_anc = num_bits - 1
    gate = cirq_ft.AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix='_a', maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    initial_state = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    final_state = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)
    num_bits = 3
    num_anc = num_bits - 1
    gate = cirq_ft.AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix='_a', maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    initial_state = [0, 0, 1, 1, 1, 1, 0, 0]
    final_state = [0, 0, 1, 1, 1, 0, 0, 0]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)
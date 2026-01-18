import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('P,n', [(v, n) for n in range(1, 4) for v in range(1 << n)])
@pytest.mark.parametrize('Q,m', [(v, n) for n in range(1, 4) for v in range(1 << n)])
@allow_deprecated_cirq_ft_use_in_tests
def test_decompose_less_than_equal_gate(P: int, n: int, Q: int, m: int):
    qubit_states = list(bit_tools.iter_bits(P, n)) + list(bit_tools.iter_bits(Q, m))
    circuit = cirq.Circuit(cirq.decompose_once(cirq_ft.LessThanEqualGate(n, m).on(*cirq.LineQubit.range(n + m + 1)), context=cirq.DecompositionContext(cirq.GreedyQubitManager(prefix='_c'))))
    qubit_order = tuple(sorted(circuit.all_qubits()))
    num_ancillas = len(circuit.all_qubits()) - n - m - 1
    initial_state = qubit_states + [0] + [0] * num_ancillas
    output_state = qubit_states + [int(P <= Q)] + [0] * num_ancillas
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, qubit_order, initial_state, output_state)
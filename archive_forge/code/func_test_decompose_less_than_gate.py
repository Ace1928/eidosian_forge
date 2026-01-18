import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('bits', [*range(8)])
@pytest.mark.parametrize('val', [3, 5, 7, 8, 9])
@allow_deprecated_cirq_ft_use_in_tests
def test_decompose_less_than_gate(bits: int, val: int):
    qubit_states = list(bit_tools.iter_bits(bits, 3))
    circuit = cirq.Circuit(cirq.decompose_once(cirq_ft.LessThanGate(3, val).on(*cirq.LineQubit.range(4))))
    if val < 8:
        initial_state = [0] * 4 + qubit_states + [0]
        output_state = [0] * 4 + qubit_states + [int(bits < val)]
    else:
        initial_state = [0]
        output_state = [1]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, sorted(circuit.all_qubits()), initial_state, output_state)
from typing import List, Sequence
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('control_val', [0, 1])
@allow_deprecated_cirq_ft_use_in_tests
def test_ising_zero_bitflip_select(control_val):
    num_sites = 4
    target_bitsize = num_sites
    num_select_unitaries = 2 * num_sites
    selection_bitsize = int(np.ceil(np.log2(num_select_unitaries)))
    all_qubits = cirq.LineQubit.range(2 * selection_bitsize + target_bitsize + 1)
    control, selection, target = (all_qubits[0], all_qubits[1:2 * selection_bitsize:2], all_qubits[2 * selection_bitsize + 1:])
    ham = get_1d_Ising_hamiltonian(target, 1, 1)
    dense_pauli_string_hamiltonian = [tt.dense(target) for tt in ham]
    op = cirq_ft.GenericSelect(selection_bitsize=selection_bitsize, target_bitsize=target_bitsize, select_unitaries=dense_pauli_string_hamiltonian, control_val=control_val).on(control, *selection, *target)
    circuit = cirq.Circuit(cirq.decompose(op))
    all_qubits = circuit.all_qubits()
    for selection_integer in range(num_select_unitaries):
        qubit_vals = {x: int(control_val) if x == control else 0 for x in all_qubits}
        qubit_vals.update(zip(selection, iter_bits(selection_integer, selection_bitsize)))
        initial_state = [qubit_vals[x] for x in all_qubits]
        for i, pauli_val in enumerate(dense_pauli_string_hamiltonian[selection_integer]):
            if pauli_val == cirq.X:
                qubit_vals[target[i]] = 1
        final_state = [qubit_vals[x] for x in all_qubits]
        cirq_ft.infra.testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)
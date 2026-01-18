import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('selection_bitsize,target_bitsize', [[3, 5], [3, 7], [4, 5]])
@allow_deprecated_cirq_ft_use_in_tests
def test_apply_gate_to_lth_qubit(selection_bitsize, target_bitsize):
    greedy_mm = cirq.GreedyQubitManager(prefix='_a', maximize_reuse=True)
    gate = cirq_ft.ApplyGateToLthQubit(cirq_ft.SelectionRegister('selection', selection_bitsize, target_bitsize), lambda _: cirq.X)
    g = cirq_ft.testing.GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    assert len(g.all_qubits) <= target_bitsize + 2 * (selection_bitsize + infra.total_bits(gate.control_registers)) - 1
    for n in range(target_bitsize):
        qubit_vals = {q: 0 for q in g.all_qubits}
        qubit_vals.update({c: 1 for c in g.quregs['control']})
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(n, selection_bitsize)))
        initial_state = [qubit_vals[x] for x in g.all_qubits]
        qubit_vals[g.quregs['target'][n]] = 1
        final_state = [qubit_vals[x] for x in g.all_qubits]
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(g.decomposed_circuit, g.all_qubits, initial_state, final_state)
import itertools
from typing import Sequence, Tuple
import cirq
import cirq_ft
import pytest
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('target_shape', [pytest.param((2, 3, 2), marks=pytest.mark.slow), (2, 2, 2)])
@allow_deprecated_cirq_ft_use_in_tests
def test_multi_dimensional_unary_iteration_gate(target_shape: Tuple[int, int, int]):
    greedy_mm = cirq.GreedyQubitManager(prefix='_a', maximize_reuse=True)
    gate = ApplyXToIJKthQubit(target_shape)
    g = cirq_ft.testing.GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    assert len(g.all_qubits) <= infra.total_bits(gate.signature) + infra.total_bits(gate.selection_registers) - 1
    max_i, max_j, max_k = target_shape
    i_len, j_len, k_len = tuple((reg.total_bits() for reg in gate.selection_registers))
    for i, j, k in itertools.product(range(max_i), range(max_j), range(max_k)):
        qubit_vals = {x: 0 for x in g.operation.qubits}
        qubit_vals.update(zip(g.quregs['i'], iter_bits(i, i_len)))
        qubit_vals.update(zip(g.quregs['j'], iter_bits(j, j_len)))
        qubit_vals.update(zip(g.quregs['k'], iter_bits(k, k_len)))
        initial_state = [qubit_vals[x] for x in g.operation.qubits]
        for reg_name, idx in zip(['t1', 't2', 't3'], [i, j, k]):
            qubit_vals[g.quregs[reg_name][idx]] = 1
        final_state = [qubit_vals[x] for x in g.operation.qubits]
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(g.circuit, g.operation.qubits, initial_state, final_state)
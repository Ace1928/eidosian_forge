from typing import List, Sequence
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_generic_select_consistent_protocols_and_controlled():
    select_bitsize, num_select, num_sites = (3, 6, 3)
    target = cirq.LineQubit.range(num_sites)
    ham = get_1d_Ising_hamiltonian(target, 1, 1)
    dps_hamiltonian = [tt.dense(target) for tt in ham]
    assert len(dps_hamiltonian) == num_select
    gate = cirq_ft.GenericSelect(select_bitsize, num_sites, dps_hamiltonian)
    op = gate.on_registers(**infra.get_named_qubits(gate.signature))
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq\nimport cirq_ft')
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(gate.controlled(), gate.controlled(num_controls=1), gate.controlled(control_values=(1,)), op.controlled_by(cirq.q('control')).gate)
    equals_tester.add_equality_group(gate.controlled(control_values=(0,)), gate.controlled(num_controls=1, control_values=(0,)), op.controlled_by(cirq.q('control'), control_values=(0,)).gate)
    with pytest.raises(NotImplementedError, match='Cannot create a controlled version'):
        _ = gate.controlled(num_controls=2)
import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_hubbard_model_consistent_protocols():
    select_gate = cirq_ft.SelectHubbard(x_dim=2, y_dim=2)
    prepare_gate = cirq_ft.PrepareHubbard(x_dim=2, y_dim=2, t=1, mu=2)
    cirq.testing.assert_equivalent_repr(select_gate, setup_code='import cirq_ft')
    cirq.testing.assert_equivalent_repr(prepare_gate, setup_code='import cirq_ft')
    select_op = select_gate.on_registers(**infra.get_named_qubits(select_gate.signature))
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(select_gate.controlled(), select_gate.controlled(num_controls=1), select_gate.controlled(control_values=(1,)), select_op.controlled_by(cirq.q('control')).gate)
    equals_tester.add_equality_group(select_gate.controlled(control_values=(0,)), select_gate.controlled(num_controls=1, control_values=(0,)), select_op.controlled_by(cirq.q('control'), control_values=(0,)).gate)
    with pytest.raises(NotImplementedError, match='Cannot create a controlled version'):
        _ = select_gate.controlled(num_controls=2)
    expected_symbols = ['U', 'V', 'p_x', 'p_y', 'alpha', 'q_x', 'q_y', 'beta']
    expected_symbols += ['target'] * 8
    expected_symbols[0] = 'SelectHubbard'
    assert cirq.circuit_diagram_info(select_gate).wire_symbols == tuple(expected_symbols)
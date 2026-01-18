from typing import Optional, Sequence, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from attr import frozen
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.algos.mean_estimation import CodeForRandomVariable, MeanEstimationOperator
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_mean_estimation_operator_consistent_protocols():
    p, selection_bitsize, y_1, target_bitsize, arctan_bitsize = (0.1, 2, 1, 1, 4)
    synthesizer = BernoulliSynthesizer(p, selection_bitsize)
    encoder = BernoulliEncoder(p, (0, y_1), selection_bitsize, target_bitsize)
    code = CodeForRandomVariable(synthesizer=synthesizer, encoder=encoder)
    mean_gate = MeanEstimationOperator(code, arctan_bitsize=arctan_bitsize)
    op = mean_gate.on_registers(**infra.get_named_qubits(mean_gate.signature))
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(mean_gate.controlled(), mean_gate.controlled(num_controls=1), mean_gate.controlled(control_values=(1,)), op.controlled_by(cirq.q('control')).gate)
    equals_tester.add_equality_group(mean_gate.controlled(control_values=(0,)), mean_gate.controlled(num_controls=1, control_values=(0,)), op.controlled_by(cirq.q('control'), control_values=(0,)).gate)
    with pytest.raises(NotImplementedError, match='Cannot create a controlled version'):
        _ = mean_gate.controlled(num_controls=2)
    assert mean_gate.with_power(5) ** 2 == MeanEstimationOperator(code, arctan_bitsize=arctan_bitsize, power=10)
    expected_symbols = ['U_ko'] * cirq.num_qubits(mean_gate)
    assert cirq.circuit_diagram_info(mean_gate).wire_symbols == tuple(expected_symbols)
    control_symbols = ['@']
    assert cirq.circuit_diagram_info(mean_gate.controlled()).wire_symbols == tuple(control_symbols + expected_symbols)
    control_symbols = ['@(0)']
    assert cirq.circuit_diagram_info(mean_gate.controlled(control_values=(0,))).wire_symbols == tuple(control_symbols + expected_symbols)
    expected_symbols[-1] = 'U_ko^2'
    assert cirq.circuit_diagram_info(mean_gate.with_power(2).controlled(control_values=(0,))).wire_symbols == tuple(control_symbols + expected_symbols)
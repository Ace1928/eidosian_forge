import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_gate_helper():
    g = cirq_ft.testing.GateHelper(cirq_ft.And(cv=(1, 0, 1, 0)))
    assert g.gate == cirq_ft.And(cv=(1, 0, 1, 0))
    assert g.r == cirq_ft.Signature([cirq_ft.Register('ctrl', bitsize=1, shape=4), cirq_ft.Register('junk', bitsize=1, shape=2, side=cirq_ft.infra.Side.RIGHT), cirq_ft.Register('target', bitsize=1, side=cirq_ft.infra.Side.RIGHT)])
    expected_quregs = {'ctrl': np.array([[cirq.q(f'ctrl[{i}]')] for i in range(4)]), 'junk': np.array([[cirq.q(f'junk[{i}]')] for i in range(2)]), 'target': [cirq.NamedQubit('target')]}
    for key in expected_quregs:
        assert np.array_equal(g.quregs[key], expected_quregs[key])
    assert g.operation.qubits == tuple(g.all_qubits)
    assert len(g.circuit) == 1
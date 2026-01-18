import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('eigen_gate_type', [cirq.XXPowGate, cirq.YYPowGate, cirq.ZZPowGate])
def test_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(eigen_gate_type)
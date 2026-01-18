import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('phase', [1, 1j, -1])
def test_gate_act_on_tableau(phase):
    original_tableau = cirq.CliffordTableau(0)
    args = cirq.CliffordTableauSimulationState(original_tableau.copy(), np.random.RandomState())
    cirq.act_on(cirq.GlobalPhaseGate(phase), args, qubits=(), allow_decompose=False)
    assert args.tableau == original_tableau
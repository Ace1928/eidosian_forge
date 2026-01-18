from typing import Any, Sequence
import numpy as np
import pytest
import cirq
from cirq.sim import simulation_state
from cirq.testing import PhaseUsingCleanAncilla, PhaseUsingDirtyAncilla
@pytest.mark.parametrize('exp', np.linspace(0, 2 * np.pi, 10))
def test_delegating_gate_channel(exp):
    q = cirq.LineQubit(0)
    test_circuit = cirq.Circuit()
    test_circuit.append(cirq.H(q))
    test_circuit.append(DelegatingAncillaZ(exp, True).on(q))
    control_circuit = cirq.Circuit(cirq.H(q))
    control_circuit.append(cirq.ZPowGate(exponent=exp).on(q))
    assert_test_circuit_for_sv_simulator(test_circuit, control_circuit)
    assert_test_circuit_for_dm_simulator(test_circuit, control_circuit)
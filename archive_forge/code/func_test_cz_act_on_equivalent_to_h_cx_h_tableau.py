import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_cz_act_on_equivalent_to_h_cx_h_tableau():
    state1 = cirq.CliffordTableauSimulationState(tableau=cirq.CliffordTableau(num_qubits=2), qubits=cirq.LineQubit.range(2), prng=np.random.RandomState())
    state2 = cirq.CliffordTableauSimulationState(tableau=cirq.CliffordTableau(num_qubits=2), qubits=cirq.LineQubit.range(2), prng=np.random.RandomState())
    cirq.act_on(cirq.S, sim_state=state1, qubits=[cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.S, sim_state=state2, qubits=[cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.H, sim_state=state1, qubits=[cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.CNOT, sim_state=state1, qubits=cirq.LineQubit.range(2), allow_decompose=False)
    cirq.act_on(cirq.H, sim_state=state1, qubits=[cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.CZ, sim_state=state2, qubits=cirq.LineQubit.range(2), allow_decompose=False)
    assert state1.tableau == state2.tableau
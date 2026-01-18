from typing import Any, Sequence
import numpy as np
import pytest
import cirq
from cirq.sim import simulation_state
from cirq.testing import PhaseUsingCleanAncilla, PhaseUsingDirtyAncilla
def test_decompose_for_gate_allocating_qubits_raises():

    class Composite(cirq.testing.SingleQubitGate):

        def _decompose_(self, qubits):
            anc = cirq.NamedQubit('anc')
            yield cirq.CNOT(*qubits, anc)
    args = ExampleSimulationState()
    with pytest.raises(TypeError, match='add_qubits but not remove_qubits'):
        simulation_state.strat_act_on_from_apply_decompose(Composite(), args, [cirq.LineQubit(0)])
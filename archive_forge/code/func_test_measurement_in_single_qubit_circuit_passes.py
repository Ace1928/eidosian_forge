from typing import Any, Dict, Optional, Sequence
import cirq
def test_measurement_in_single_qubit_circuit_passes():
    state = create_container([q0])
    assert len(set(state.values())) == 2
    state.apply_operation(cirq.measure(q0))
    assert len(set(state.values())) == 2
    assert state[q0] is not state[None]
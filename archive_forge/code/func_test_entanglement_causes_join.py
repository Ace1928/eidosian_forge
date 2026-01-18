from typing import Any, Dict, Optional, Sequence
import cirq
def test_entanglement_causes_join():
    state = create_container(qs2)
    assert len(set(state.values())) == 3
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 2
    assert state[q0] is state[q1]
    assert state[None] is not state[q0]
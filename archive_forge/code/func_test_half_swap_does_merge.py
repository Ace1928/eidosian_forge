from typing import Any, Dict, Optional, Sequence
import cirq
def test_half_swap_does_merge():
    state = create_container(qs2)
    state.apply_operation(cirq.SWAP(q0, q1) ** 0.5)
    assert len(set(state.values())) == 2
    assert state[q0] is state[q1]
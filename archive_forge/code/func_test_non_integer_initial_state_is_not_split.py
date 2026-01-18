import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_non_integer_initial_state_is_not_split():
    sim = SplittableCountingSimulator()
    state = sim._create_simulation_state(entangled_state_repr, (q0, q1))
    assert len(set(state.values())) == 2
    assert (state[q0].data == entangled_state_repr).all()
    assert state[q1] is state[q0]
    assert state[None].data == 0
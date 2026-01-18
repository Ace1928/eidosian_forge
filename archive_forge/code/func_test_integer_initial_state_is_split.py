import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_integer_initial_state_is_split():
    sim = SplittableCountingSimulator()
    state = sim._create_simulation_state(2, (q0, q1))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0].data == 1
    assert state[q1].data == 0
    assert state[None].data == 0
import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_integer_initial_state_is_not_split_if_impossible():
    sim = CountingSimulator()
    state = sim._create_simulation_state(2, (q0, q1))
    assert isinstance(state, CountingSimulationState)
    assert not isinstance(state, SplittableCountingSimulationState)
    assert state[q0] is state[q1]
    assert state.data == 2
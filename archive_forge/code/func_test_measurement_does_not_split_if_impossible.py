import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_measurement_does_not_split_if_impossible():
    sim = CountingSimulator()
    state = sim._create_simulation_state(2, (q0, q1))
    assert isinstance(state, CountingSimulationState)
    assert not isinstance(state, SplittableCountingSimulationState)
    state.apply_operation(cirq.measure(q0))
    assert isinstance(state, CountingSimulationState)
    assert not isinstance(state, SplittableCountingSimulationState)
    assert state[q0] is state[q1]
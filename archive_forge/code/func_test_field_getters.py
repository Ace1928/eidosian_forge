from typing import Any, Sequence
import numpy as np
import pytest
import cirq
from cirq.sim import simulation_state
from cirq.testing import PhaseUsingCleanAncilla, PhaseUsingDirtyAncilla
def test_field_getters():
    args = ExampleSimulationState()
    assert args.prng is np.random
    assert args.qubit_map == {q: i for i, q in enumerate(cirq.LineQubit.range(2))}